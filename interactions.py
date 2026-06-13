"""Per-patient interaction store: one JSON file per phone number under ./interactions.

Each file accumulates everything we know about a patient across calls:

  {
    "phone": "+15551234567",
    "sessions": [                      # one per call (triage or check-in)
      {
        "call_id": "abc123",
        "agent": "triage",
        "started_at": "2026-06-13T...Z",
        "turns": [                     # structured triage steps (running memory)
          {"at": ..., "descriptions": [...], "empathy": "...",
           "candidates": ["MDD", ...], "next_question": "...",
           "future_checkin": ["..."]}
        ],
        "transcript": [{"role": "caller"|"agent", "text": "..."}]
      }
    ],
    "pending_checkins": ["..."]         # questions run_checkin.py asks later
  }

`candidates` per turn IS the running memory of the differential over the call.
"""
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("interactions")

DIR = Path("interactions")


def _safe(phone: str) -> str:
    """Filename-safe key from a phone number (digits only; 'unknown' if absent)."""
    return re.sub(r"\D", "", phone or "") or "unknown"


def path_for(phone: str) -> Path:
    return DIR / f"{_safe(phone)}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(phone: str) -> dict:
    """Load the patient's record, or a fresh skeleton if none exists."""
    p = path_for(phone)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Corrupt interaction file %s; starting fresh", p)
    return {"phone": phone or "unknown", "sessions": [], "pending_checkins": []}


def save(record: dict) -> None:
    DIR.mkdir(exist_ok=True)
    path_for(record.get("phone")).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def start_session(record: dict, call_id: str, agent: str) -> dict:
    """Append a new call session to the record and return it."""
    session = {
        "call_id": call_id,
        "agent": agent,
        "started_at": _now(),
        "turns": [],
        "transcript": [],
    }
    record.setdefault("sessions", []).append(session)
    return session


def add_turn(record: dict, session: dict, descriptions, empathy,
             candidates, next_question, future_checkin) -> None:
    """Record one triage step (symptom(s) -> differential -> next question)."""
    session["turns"].append({
        "at": _now(),
        "descriptions": list(descriptions or []),
        "empathy": empathy or "",
        "candidates": list(candidates or []),
        "next_question": next_question or "",
        "future_checkin": list(future_checkin or []),
    })
    # Roll new future check-in questions into the patient-level pending list
    # (deduped) so run_checkin.py can ask them on a later call.
    pending = record.setdefault("pending_checkins", [])
    for q in future_checkin or []:
        if q and q not in pending:
            pending.append(q)


def add_transcript(session: dict, role: str, text: str) -> None:
    if text:
        session.setdefault("transcript", []).append({"role": role, "text": text})


def pending_checkins(phone: str) -> list:
    return load(phone).get("pending_checkins", [])


def clear_pending(record: dict) -> None:
    record["pending_checkins"] = []
