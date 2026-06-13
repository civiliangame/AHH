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


# ---------------------------------------------------------------------------
# 1. Schemas advertised to the model (goes into session.update -> "tools")
# ---------------------------------------------------------------------------
TRIAGE_TOOLS = [
    {
        "type": "function",
        "name": "assess_suicide_risk",
        "description": (
            "Score suicide risk from C-SSRS answers. Call this the moment any "
            "suicidal ideation, intent, or plan is mentioned — before anything else."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ideation": {"type": "boolean", "description": "Any wish to be dead or suicidal thoughts"},
                "plan":     {"type": "boolean", "description": "Has a specific method or plan"},
                "intent":   {"type": "boolean", "description": "Intends to act on it"},
                "means":    {"type": "boolean", "description": "Has access to means"},
                "verbatim": {"type": "string",  "description": "What the caller actually said"},
            },
            "required": ["ideation"],
        },
    },
    {
        "type": "function",
        "name": "score_assessment",
        "description": (
            "Compute the validated score and severity tier for a completed "
            "screening instrument. Always use this — never total the items yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instrument": {"type": "string", "enum": ["PHQ-9", "GAD-7"]},
                "responses":  {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Item scores in order (each 0-3)",
                },
            },
            "required": ["instrument", "responses"],
        },
    },
    {
        "type": "function",
        "name": "save_triage_summary",
        "description": (
            "Persist the final structured triage result and recommended level "
            "of care. Call once at the end of intake."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domains":   {"type": "array", "items": {"type": "string"},
                              "description": "Symptom domains that screened positive"},
                "level_of_care": {"type": "string",
                                  "enum": ["crisis", "urgent", "routine", "self_help"]},
                "notes":     {"type": "string"},
            },
            "required": ["level_of_care"],
        },
    },
]


# ---------------------------------------------------------------------------
# 2. Implementations — one function per tool name above
# ---------------------------------------------------------------------------
def assess_suicide_risk(ideation=False, plan=False, intent=False,
                        means=False, verbatim=""):
    print("OH NO")
    if intent or (plan and means):
        risk = "high"
    elif plan or intent:
        risk = "moderate"
    elif ideation:
        risk = "low"
    else:
        risk = "none"
    log.warning("Suicide risk assessed: %s | %r", risk, verbatim)
    return {"risk": risk, "escalate": risk in ("moderate", "high")}


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
    "assess_suicide_risk": assess_suicide_risk,
    "score_assessment": score_assessment,
    "save_triage_summary": save_triage_summary,
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
