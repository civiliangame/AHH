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

import httpx

import config

log = logging.getLogger("triage")

global_descriptions = []
latest_candidates = []  # candidate conditions from the most recent differential

# ---------------------------------------------------------------------------
# 1. Schemas advertised to the model (goes into session.update -> "tools")
# ---------------------------------------------------------------------------
TRIAGE_TOOLS = [
    {
        "type": "function",
        "name": "recordSymptom",
        "description": (
            "Record a symptom mentioned by the patient. Call this the moment any "
            "symptom is mentioned — before anything else. Be very detailed with your "
            "description. Do NOT speak when you call this — the system does the talking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Detailed description of the symptom"},
                "empathy": {"type": "string", "description": "One short, warm sentence acknowledging how the patient feels. The SYSTEM speaks this to the patient automatically — do NOT say it yourself."},
            },
            "required": ["description", "empathy"],
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
                "description": "ONE short, spoken-friendly question that best narrows the differential.",
            },
        },
        "required": ["candidates", "next_question"],
        "additionalProperties": False,
    },
}

_DIFFERENTIAL_SYSTEM = (
    "You are a clinical triage reasoning engine for a mental-health intake line. "
    "Given the patient's reported symptoms so far, use the DSM-5 reference "
    "collection (file_search) to identify the most likely candidate conditions, "
    "then propose ONE short, spoken-friendly question that best narrows the "
    "differential. You are NOT diagnosing — you are guiding intake. "
    "Respond with JSON only."
)

# Safe fallback so the voice flow never stalls if the API call fails.
_FALLBACK = {
    "candidates": [],
    "next_question": "Can you tell me a bit more about how you've been feeling lately?",
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


def get_next_question():
    """Differential step: feed all recorded symptoms + the DSM-5 collection to a
    fast Grok model and return {"candidates": [...], "next_question": "..."}.

    Uses the Responses API (/v1/responses) because file_search over a collection
    is NOT supported on chat/completions. Sync on purpose — server.py runs it via
    asyncio.to_thread so its latency overlaps the empathy line instead of
    blocking the audio relay. Always returns a dict with next_question; falls
    back on any error.
    """
    symptoms = list(global_descriptions)
    if not symptoms:
        return dict(_FALLBACK)

    bullets = "\n".join(f"- {s}" for s in symptoms)
    payload = {
        "model": config.GROK_FAST_MODEL,
        "instructions": _DIFFERENTIAL_SYSTEM,
        "input": (
            f"Patient-reported symptoms so far:\n{bullets}\n\n"
            "List the candidate conditions and ONE narrowing question."
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
        if not result.get("next_question"):
            result["next_question"] = _FALLBACK["next_question"]
        global latest_candidates
        latest_candidates = result["candidates"]
        log.info("Differential: candidates=%s next_q=%r",
                 result["candidates"], result["next_question"])
        return result
    except Exception:
        log.exception("get_next_question failed; using fallback question")
        return dict(_FALLBACK)

# ---------------------------------------------------------------------------
# 2. Implementations — one function per tool name above
# ---------------------------------------------------------------------------
def recordSymptom(description, empathy):
    print("OH NO")
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
