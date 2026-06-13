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
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Most likely candidate mental health conditions given the symptoms so far.",
            },
            "next_question": {
                "type": "string",
                "description": "ONE short, spoken-friendly question that best narrows the differential. Must NOT repeat any already-asked question. Empty string if done.",
            },
            "done": {
                "type": "boolean",
                "description": "true if there is enough information for an initial impression, OR no further question would meaningfully narrow the differential.",
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
                        "evidence": {"type": "string", "description": "What the patient said that supports this status (or why it's unclear)."},
                    },
                    "required": ["condition", "criterion", "status", "evidence"],
                    "additionalProperties": False,
                },
            },
            "future_checkin": {
                "type": "array",
                "description": (
                    "Zero or more follow-up questions worth asking in a LATER call "
                    "to track how the patient changes over time (sleep, mood "
                    "trajectory, treatment response) — for longitudinal accuracy, "
                    "not this call. Empty array if none are warranted."
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
        "required": ["candidates", "next_question", "done", "dsm5_criteria", "future_checkin"],
        "additionalProperties": False,
    },
}

_DIFFERENTIAL_SYSTEM = (
    "You are a clinical triage reasoning engine for a mental-health intake line. "
    "Given the patient's reported symptoms so far, use the DSM-5 reference "
    "collection (file_search) to identify the most likely candidate conditions "
    "and to build a profile: for the relevant DSM-5 criteria, mark each as met / "
    "not_met / unclear with the patient's own words as evidence (`dsm5_criteria`). "
    "Re-assess the criteria against ALL symptoms every turn — the profile grows as "
    "the patient answers. Propose ONE short, spoken-friendly question that best "
    "narrows the differential or resolves an `unclear` criterion. Also propose "
    "`future_checkin`: zero or more {days, message} follow-ups to ask on a LATER "
    "call to track change over time (set days to when it should be asked, e.g. 3, "
    "7, 14). If earlier candidate conditions are provided, refine that running "
    "differential rather than starting over. Do NOT repeat any question already "
    "asked (the list is provided) — pick a genuinely new angle. Set `done` to true "
    "(and leave `next_question` empty) when you have enough for an initial "
    "impression or no further question would meaningfully help. You are NOT "
    "diagnosing — you are guiding intake. Respond with JSON only."
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


def get_next_question(symptoms, prior_candidates=None, asked_questions=None, phone=None):
    """Differential step: feed the recorded symptoms (+ the running list of prior
    candidates) and the DSM-5 collection to a fast Grok model. Returns
    {"candidates": [...], "next_question": "...", "future_checkin": [...]}.

    `symptoms` is the per-call list of everything recordSymptom has logged.
    `prior_candidates` is the differential from earlier turns, so the model
    refines its running memory instead of starting over.

    Uses the Responses API (/v1/responses) because file_search over a collection
    is NOT supported on chat/completions. Sync on purpose — server.py runs it via
    asyncio.to_thread so its latency overlaps the empathy line instead of
    blocking the audio relay. Always returns a dict with next_question; falls
    back on any error.
    """
    symptoms = list(symptoms or [])
    if not symptoms:
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
    payload = {
        "model": config.GROK_TRIAGE_MODEL,
        # Disable reasoning ("none") — grok-4.3 answers directly, lower latency.
        "reasoning": {"effort": "none"},
        "instructions": _DIFFERENTIAL_SYSTEM,
        "input": (
            f"Patient-reported symptoms so far:\n{bullets}{prior}{asked}\n\n"
            "List the candidate conditions, ONE new narrowing question (or set "
            "done=true with an empty question if no more are needed), and any "
            "future check-in questions."
        ),
        # Server-side tool: xAI searches the DSM-5 collection and grounds the
        # answer. file_search lives on the Responses API, not chat/completions.
        "tools": [{
            "type": "file_search",
            "vector_store_ids": [config.DSM5_COLLECTION_ID],
        }],
        "text": {"format": {"type": "json_schema", **_NEXT_STEP_SCHEMA}},
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {config.XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{config.XAI_API_BASE}/responses"
    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = json.loads(_extract_output_text(resp.json()))
        result.setdefault("candidates", [])
        result.setdefault("future_checkin", [])
        result.setdefault("dsm5_criteria", [])
        result.setdefault("done", False)
        # Only backfill a question when not done — an empty question + done=true
        # is the "close the call" signal and must be preserved.
        if not result.get("done") and not result.get("next_question"):
            result["next_question"] = _FALLBACK["next_question"]
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
