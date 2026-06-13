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
import logging

log = logging.getLogger("triage")
import time

# ---------------------------------------------------------------------------
# 1. Schemas advertised to the model (goes into session.update -> "tools")
# ---------------------------------------------------------------------------
TRIAGE_TOOLS = [
    {
        "type": "function",
        "name": "recordSymptom",
        "description": (
            "Record a symptom mentioned by the patient. Call this the moment any "
            "symptom is mentioned — before anything else. Be very detailed with your description"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Detailed description of the symptom"},
                "empathy": {"type": "string", "description": "Empathetic response to the patient to be read aloud, word for word. Should be one sentence."},
            },
            "required": ["description", "empathy"],
        },
    }
]


def get_next_question():
    time.sleep(3)
    return {"status": "success", "next_question": "When did this start?"}

# ---------------------------------------------------------------------------
# 2. Implementations — one function per tool name above
# ---------------------------------------------------------------------------
def recordSymptom(description, empathy):
    print("OH NO")
    print(f"Recording symptom: {description}")
    return {"status": "success", "message": "Ask next: When did this start?"}


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
