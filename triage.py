"""Function tools for the DSM-5 triage voice agent.

Two halves live here, side by side so they never drift apart:

  1. TRIAGE_TOOLS  — the JSON schemas advertised to xAI in `session.update`.
                     This is what the model "sees" and decides to call.
  2. handle()      — the Python that actually runs when the model calls a tool,
                     dispatched by name. Add a function, add it to HANDLERS,
                     add its schema to TRIAGE_TOOLS. That's the whole loop.

The model never does arithmetic or risk logic itself — it gathers the answers
and calls these functions, which return structured results it then speaks.
"""
import json
import logging
import os
import re
import subprocess
import sys
import tempfile

import httpx

import config

log = logging.getLogger("triage")

global_descriptions = []
latest_candidates = []  # candidate conditions from the most recent differential

# Stop asking after this many questions, even if the differential isn't settled.
MAX_QUESTIONS = 5

# Spoken verbatim (force_message) when triage is complete.
CLOSING_LINE = (
    "I think I have everything I need to get this started. "
    "I'll call you in a few days to follow up and see how you're doing."
)

# ---------------------------------------------------------------------------
# 1. Schemas advertised to the model (goes into session.update -> "tools")
# ---------------------------------------------------------------------------
TRIAGE_TOOLS = [
    {
        "type": "function",
        "name": "recordSymptom",
        "description": (
            "Record a symptom mentioned by the patient. Call this the moment any "
            "symptom is mentioned, in the SAME turn as your one-sentence spoken "
            "empathy. Be very detailed with your description."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Detailed description of the symptom"},
            },
            "required": ["description"],
        },
    }
]


# Structured-output contract the model must return.
_NEXT_STEP_SCHEMA = {
    "name": "triage_step",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            # next_question and done are declared first on purpose: the model
            # emits properties in schema order, so streaming can surface the
            # spoken question + done flag before the bulkier fields below finish.
            "next_question": {
                "type": "string",
                "description": "ONE short, spoken-friendly question that best narrows the differential. Must NOT repeat any already-asked question. Empty string if done.",
            },
            "done": {
                "type": "boolean",
                "description": "true if there is enough information for an initial impression, OR no further question would meaningfully narrow the differential.",
            },
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Most likely candidate mental health conditions given the symptoms so far.",
            },
            "dsm5_criteria": {
                "type": "array",
                "description": (
                    "The specific DSM-5 criteria relevant to the candidate "
                    "conditions and whether the patient's reports meet them. "
                    "Build this up as the patient answers — re-assess every turn "
                    "against ALL symptoms reported so far."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "condition": {"type": "string", "description": "DSM-5 condition, e.g. 'Major Depressive Disorder'."},
                        "criterion": {"type": "string", "description": "The specific criterion, e.g. 'A1: Depressed mood most of the day, nearly every day'."},
                        "status": {"type": "string", "enum": ["met", "not_met", "unclear"], "description": "Whether the patient's reports meet this criterion."},
                    },
                    "required": ["condition", "criterion", "status"],
                    "additionalProperties": False,
                },
            },
            "future_checkin": {
                "type": "array",
                "description": (
                    "Follow-up questions to ask on a LATER call to track how the "
                    "patient changes over time (sleep, mood trajectory, treatment "
                    "response) — for longitudinal accuracy, not this call. MUST be "
                    "an empty array unless `done` is true: only populate this on the "
                    "final turn. Empty array if none are warranted."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "description": "How many days from now to ask this (e.g. 3, 7, 14)."},
                        "message": {"type": "string", "description": "The exact follow-up question to ask on the check-in call."},
                    },
                    "required": ["days", "message"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["next_question", "done", "candidates", "dsm5_criteria", "future_checkin"],
        "additionalProperties": False,
    },
}

# DSM-5 reference material, loaded ONCE into memory at import (process start) and
# held for the life of the server, so every call/turn reuses it with zero disk or
# network cost. This replaces the xAI file_search vector store: instead of a
# server-side retrieval round-trip per turn, the (small) reference text is inlined
# into the model's instructions — which are constant, so xAI can prompt-cache them.
_DSM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsm.txt")


def _load_dsm_reference():
    try:
        with open(_DSM_PATH, encoding="utf-8") as f:
            text = f.read().strip()
        log.info("Loaded DSM-5 reference from %s (%d chars)", _DSM_PATH, len(text))
        return text
    except Exception:
        log.exception("could not load %s — differential will run UNGROUNDED", _DSM_PATH)
        return ""


_DSM_REFERENCE = _load_dsm_reference()

_DIFFERENTIAL_SYSTEM = (
    "You are a clinical triage reasoning engine for a mental-health intake line. "
    "Given the patient's reported symptoms so far, use ONLY the DSM-5 reference "
    "material provided below to identify the most likely candidate conditions "
    "and to build a profile: for the relevant DSM-5 criteria, mark each as met / "
    "not_met / unclear (`dsm5_criteria`). "
    "Re-assess the criteria against ALL symptoms every turn — the profile grows as "
    "the patient answers. Propose ONE short, spoken-friendly question that best "
    "narrows the differential or resolves an `unclear` criterion. ONLY when you "
    "set `done` to true (the final turn) propose `future_checkin`: {days, message} "
    "follow-ups to ask on a LATER call to track change over time (set days to when "
    "it should be asked, e.g. 3, 7, 14). On every non-final turn, return an empty "
    "`future_checkin` array. If earlier candidate conditions are provided, refine that running "
    "differential rather than starting over. Do NOT repeat any question already "
    "asked (the list is provided) — pick a genuinely new angle. Set `done` to true "
    "(and leave `next_question` empty) when you have enough for an initial "
    "impression or no further question would meaningfully help. You are NOT "
    "diagnosing — you are guiding intake. Respond with JSON only."
    "\n\n===== DSM-5 REFERENCE MATERIAL =====\n" + _DSM_REFERENCE
)

# Safe fallback so the voice flow never stalls if the API call fails.
_FALLBACK = {
    "candidates": [],
    "next_question": "Can you tell me a bit more about how you've been feeling lately?",
    "done": False,
    "dsm5_criteria": [],
    "future_checkin": [],
}


def _extract_output_text(data: dict) -> str:
    """Pull the model's text out of a Responses API payload.

    Prefers the convenience `output_text`; otherwise walks `output` for the
    message item (skipping the reasoning item) and returns its text content.
    """
    if data.get("output_text"):
        return data["output_text"]
    for item in data.get("output", []):
        if item.get("type") == "message":
            for chunk in item.get("content", []):
                if chunk.get("text"):
                    return chunk["text"]
    raise ValueError("no message text in Responses API output")


def _spawn_criteria_worker(phone, symptoms, candidates):
    """Fire-and-forget background PROCESS that builds the EXACT DSM-5 criteria
    profile (yes/no/need_psychiatrist/need_checkin) and writes it to a separate
    file. Decoupled from the latency-sensitive next_question path — we never wait
    on it. Inputs are passed via a temp file the worker reads then deletes.
    """
    try:
        fd, path = tempfile.mkstemp(prefix="criteria_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"phone": phone, "symptoms": symptoms,
                       "candidates": candidates}, f)
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "criteria_worker.py")
        subprocess.Popen([sys.executable, worker, path])  # detached; don't wait
        log.info("Spawned criteria_worker for %s (%d symptom(s))", phone, len(symptoms))
    except Exception:
        log.exception("could not spawn criteria_worker")


# Cheap partial-JSON extractors used while streaming, before the full object is
# parseable. next_question/done are emitted first (see _NEXT_STEP_SCHEMA), so once
# `done` appears the question string is already complete.
_NEXT_Q_RE = re.compile(r'"next_question"\s*:\s*"((?:[^"\\]|\\.)*)"')
_DONE_RE = re.compile(r'"done"\s*:\s*(true|false)')


async def get_next_question(symptoms, prior_candidates=None, asked_questions=None,
                            questions_asked=0, max_questions=None, ready=None, phone=None):
    """Differential step: feed the recorded symptoms (+ the running list of prior
    candidates) and the DSM-5 collection to a fast Grok model. Returns the full
    {"next_question", "done", "candidates", "dsm5_criteria", "future_checkin"} dict.

    `symptoms` is the per-call list of everything recordSymptom has logged.
    `prior_candidates` is the differential from earlier turns, so the model
    refines its running memory instead of starting over. `questions_asked` /
    `max_questions` tell the model how much budget is left so it can wrap up
    (set done=true, emit future_checkin) on the final allowed turn.

    Streams the Responses API (/v1/responses) — file_search over a collection is
    NOT supported on chat/completions. If `ready` (an asyncio.Future) is given, it
    is resolved with (next_question, done) the moment those two fields finish
    streaming, BEFORE the bulkier candidates/dsm5_criteria/future_checkin arrive —
    so the caller can speak the question without waiting for the whole object.
    Always returns a dict with next_question; falls back on any error.
    """
    symptoms = list(symptoms or [])
    max_questions = max_questions or MAX_QUESTIONS
    if not symptoms:
        if ready is not None and not ready.done():
            ready.set_result((_FALLBACK["next_question"], False))
        return dict(_FALLBACK)

    bullets = "\n".join(f"- {s}" for s in symptoms)
    prior = ""
    if prior_candidates:
        prior = "\n\nCandidate conditions considered in earlier turns: " + \
            ", ".join(prior_candidates)
    asked = ""
    if asked_questions:
        asked = "\n\nQuestions already asked (do NOT repeat these): " + \
            "; ".join(asked_questions)
    # Let the model see how much budget is left so the final turn is deliberate
    # (done=true + future_checkin) rather than cut off by the server's hard cap.
    this_q = questions_asked + 1
    if this_q >= max_questions:
        budget = (
            f"\n\nThis is your FINAL turn (question {this_q} of at most "
            f"{max_questions}); no more questions can be asked after this. You MUST "
            "set done=true, leave next_question empty, and provide future_checkin."
        )
    else:
        budget = (
            f"\n\nThis is question {this_q} of at most {max_questions}. Set "
            "done=true now if you already have enough for an initial impression."
        )
    payload = {
        "model": config.GROK_TRIAGE_MODEL,
        # Disable reasoning ("none") — grok-4.3 answers directly, lower latency.
        "reasoning": {"effort": "none"},
        "stream": True,
        "instructions": _DIFFERENTIAL_SYSTEM,
        "input": (
            f"Patient-reported symptoms so far:\n{bullets}{prior}{asked}{budget}\n\n"
            "List the candidate conditions and ONE new narrowing question (or set "
            "done=true with an empty question if no more are needed). Only include "
            "future check-in questions if you set done=true."
        ),
        # No file_search tool: the DSM-5 reference is inlined into the
        # instructions above (see _DSM_REFERENCE), so the model grounds its
        # answer directly from in-memory text — no vector-store retrieval round
        # trip per turn.
        "text": {"format": {"type": "json_schema", **_NEXT_STEP_SCHEMA}},
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {config.XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{config.XAI_API_BASE}/responses"

    buf = []
    signaled = False

    def _maybe_signal():
        # As soon as both next_question and done have fully streamed, hand the
        # caller the spoken question without waiting for the rest of the object.
        nonlocal signaled
        if signaled or ready is None or ready.done():
            return
        text = "".join(buf)
        dm = _DONE_RE.search(text)
        if dm is None:
            return
        qm = _NEXT_Q_RE.search(text)
        if qm is None:
            return
        try:
            next_q = json.loads(f'"{qm.group(1)}"')
        except Exception:
            next_q = ""
        done = dm.group(1) == "true"
        if not done and not next_q:
            next_q = _FALLBACK["next_question"]
        ready.set_result((next_q, done))
        signaled = True

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        evt = json.loads(data)
                    except Exception:
                        continue
                    if evt.get("type") == "response.output_text.delta":
                        buf.append(evt.get("delta", ""))
                        _maybe_signal()

        result = json.loads("".join(buf))
        result.setdefault("candidates", [])
        result.setdefault("future_checkin", [])
        result.setdefault("dsm5_criteria", [])
        result.setdefault("done", False)
        # Only backfill a question when not done — an empty question + done=true
        # is the "close the call" signal and must be preserved.
        if not result.get("done") and not result.get("next_question"):
            result["next_question"] = _FALLBACK["next_question"]
        # If parsing finished before _maybe_signal fired (e.g. tiny response),
        # still hand the caller the answer.
        if ready is not None and not ready.done():
            ready.set_result((result["next_question"], result["done"]))
        global latest_candidates
        latest_candidates = result["candidates"]
        log.info("Differential: candidates=%s next_q=%r",
                 result["candidates"], result["next_question"])
        # Kick off the deep EXACT-criteria profiling in a separate background
        # process. It runs independently and persists to <phone>.dsm5.json — we
        # do NOT wait on it, so it adds zero latency to the next question.
        _spawn_criteria_worker(phone, symptoms, result["candidates"])
        return result
    except Exception:
        log.exception("get_next_question failed; using fallback question")
        if ready is not None and not ready.done():
            ready.set_result((_FALLBACK["next_question"], False))
        return dict(_FALLBACK)

# ---------------------------------------------------------------------------
# 2. Implementations — one function per tool name above
# ---------------------------------------------------------------------------
def recordSymptom(description):
    print(f"Recording symptom: {description}")
    global_descriptions.append(description)
    # return {"status": "success", "message": "Ask next: When did this start?"}


# PHQ-9 / GAD-7 severity bands (sum of item scores).
_BANDS = {
    "PHQ-9": [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"),
              (15, 19, "moderately severe"), (20, 27, "severe")],
    "GAD-7": [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"),
              (15, 21, "severe")],
}


def score_assessment(instrument, responses):
    total = sum(responses)
    severity = next((label for lo, hi, label in _BANDS.get(instrument, [])
                     if lo <= total <= hi), "unknown")
    return {"instrument": instrument, "total": total, "severity": severity}


def save_triage_summary(level_of_care, domains=None, notes=""):
    summary = {"level_of_care": level_of_care, "domains": domains or [], "notes": notes}
    log.info("Triage summary: %s", summary)
    # TODO: persist to a DB / EHR / queue a clinician handoff here.
    return {"saved": True, **summary}


HANDLERS = {
    "recordSymptom": recordSymptom
}


# ---------------------------------------------------------------------------
# 3. Dispatcher — called by the bridge with the model's tool-call event
# ---------------------------------------------------------------------------
def handle(name: str, arguments: dict) -> dict:
    """Run the named tool. Returns a JSON-serializable result for the model."""
    fn = HANDLERS.get(name)
    if fn is None:
        log.error("Unknown tool: %s", name)
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**arguments)
    except Exception as e:
        log.exception("Tool %s failed", name)
        return {"error": str(e)}
