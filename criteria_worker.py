"""Background DSM-5 criteria profiler — spawned (fire-and-forget) by
triage.get_next_question on every turn.

It takes the patient's full symptom list + candidate conditions and asks Grok
(grok-4.3 + the DSM-5 collection via file_search) to lay out the EXACT DSM-5
criteria for each candidate and classify whether the patient meets each:

  yes               - clearly met from what the patient reported
  no                - clearly not met
  need_psychiatrist - the DSM requires clinical judgment a phone intake can't make
  need_checkin      - can't tell yet; needs a follow-up over time (e.g. a
                      duration/persistence criterion not yet observable)

The result is written to interactions/<phone>.dsm5.json — a SEPARATE file from
the main record, so this process never races the server's record writes.

Runs as its own OS process so the (slow, thorough) criteria analysis never adds
latency to the voice loop's next question.

Usage (spawned, not run by hand):
    python criteria_worker.py <input_json_path>
where the input file is {"phone": ..., "symptoms": [...], "candidates": [...]}.
"""
import json
import logging
import os
import sys

import httpx

import config
import interactions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s criteria_worker: %(message)s",
)
log = logging.getLogger("criteria_worker")


# Structured-output contract: EXACT criteria per condition, 4-way status.
_PROFILE_SCHEMA = {
    "name": "dsm5_profile",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "condition": {"type": "string", "description": "DSM-5 condition name."},
                        "criteria": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "criterion": {"type": "string", "description": "The EXACT DSM-5 criterion text/label, e.g. 'A1: Depressed mood most of the day, nearly every day'."},
                                    "status": {"type": "string", "enum": ["yes", "no", "need_psychiatrist", "need_checkin"]},
                                    "evidence": {"type": "string", "description": "The patient's own words / the reasoning behind this status."},
                                },
                                "required": ["criterion", "status", "evidence"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["condition", "criteria"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["conditions"],
        "additionalProperties": False,
    },
}

_SYSTEM = (
    "You are a DSM-5 criteria auditor for a mental-health intake line. For EACH "
    "candidate condition, use the DSM-5 reference collection (file_search) to list "
    "its EXACT criteria, then classify whether the patient meets each based ONLY "
    "on what they have reported so far:\n"
    "  yes               - clearly met\n"
    "  no                - clearly not met\n"
    "  need_psychiatrist - the DSM requires clinical judgment a phone intake cannot make\n"
    "  need_checkin      - cannot determine yet; needs a follow-up over time (e.g. a "
    "duration or persistence criterion not yet observable)\n"
    "Quote the patient's own words as evidence. Respond with JSON only."
)


def _extract_output_text(data: dict) -> str:
    """Pull the model's text out of a Responses API payload (skip reasoning item)."""
    if data.get("output_text"):
        return data["output_text"]
    for item in data.get("output", []):
        if item.get("type") == "message":
            for chunk in item.get("content", []):
                if chunk.get("text"):
                    return chunk["text"]
    raise ValueError("no message text in Responses API output")


def build_profile(symptoms, candidates) -> list:
    bullets = "\n".join(f"- {s}" for s in symptoms)
    cond = ", ".join(candidates) if candidates else "(derive the most likely conditions yourself)"
    payload = {
        "model": config.GROK_TRIAGE_MODEL,
        "reasoning": {"effort": "none"},
        "instructions": _SYSTEM,
        "input": (
            f"Candidate conditions: {cond}\n\n"
            f"Patient-reported symptoms so far:\n{bullets}\n\n"
            "List each condition's EXACT DSM-5 criteria with a status and evidence."
        ),
        "tools": [{"type": "file_search", "vector_store_ids": [config.DSM5_COLLECTION_ID]}],
        "text": {"format": {"type": "json_schema", **_PROFILE_SCHEMA}},
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {config.XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = httpx.post(f"{config.XAI_API_BASE}/responses", headers=headers,
                      json=payload, timeout=90)
    resp.raise_for_status()
    return json.loads(_extract_output_text(resp.json())).get("conditions", [])


def main():
    if len(sys.argv) < 2:
        log.error("usage: criteria_worker.py <input_json_path>")
        return
    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8") as f:
            inp = json.load(f)
    except Exception:
        log.exception("could not read input file %s", path)
        return
    finally:
        try:
            os.unlink(path)  # clean up the temp input file
        except OSError:
            pass

    phone = inp.get("phone")
    symptoms = inp.get("symptoms", [])
    candidates = inp.get("candidates", [])
    if not symptoms:
        return
    try:
        conditions = build_profile(symptoms, candidates)
    except Exception:
        log.exception("criteria profiling failed")
        return
    interactions.save_dsm5_profile(phone, conditions)
    log.info("Saved DSM-5 profile for %s: %d condition(s)", phone, len(conditions))


if __name__ == "__main__":
    main()
