"""Per-patient interaction store: one JSON file per phone number under ./interactions.

Each file accumulates everything we know about a patient across calls:

  {
    "phone": "+15551234567",
    "profile": {                       # rolling profile, updated every turn
      "updated_at": "...Z",
      "candidates": ["MDD", ...],
      "dsm5_criteria": [
        {"condition": "Major Depressive Disorder",
         "criterion": "A1: Depressed mood most of the day",
         "status": "met"|"not_met"|"unclear"}
      ]
    },
    "sessions": [                      # one per call (triage or check-in)
      {
        "call_id": "abc123",
        "agent": "triage",
        "started_at": "2026-06-13T...Z",
        "turns": [                     # structured triage steps (running memory)
          {"at": ..., "descriptions": [...], "empathy": "...",
           "candidates": ["MDD", ...], "dsm5_criteria": [...],
           "next_question": "...", "future_checkin": [{"days": 7, "message": "..."}]}
        ],
        "transcript": [{"role": "caller"|"agent", "text": "..."}]
      }
    ],
    "pending_checkins": [{"days": 7, "message": "..."}]   # run_checkin.py asks later
  }

`candidates` per turn IS the running memory of the differential over the call.
"""
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("interactions")

DIR = Path("interactions")


def _safe(phone: str) -> str:
    """Filename-safe key from a phone number (digits only; 'unknown' if absent)."""
    return re.sub(r"\D", "", phone or "") or "unknown"


def path_for(phone: str) -> Path:
    return DIR / f"{_safe(phone)}.json"


def profile_path_for(phone: str) -> Path:
    """Separate file for the background DSM-5 criteria profile, so criteria_worker.py
    (its own process) never races the server's writes to the main record."""
    return DIR / f"{_safe(phone)}.dsm5.json"


def save_dsm5_profile(phone: str, conditions: list) -> None:
    """Write the EXACT DSM-5 criteria profile (yes/no/need_psychiatrist/need_checkin).
    Atomic (temp + os.replace) so a concurrent reader never sees a half-written file."""
    DIR.mkdir(exist_ok=True)
    data = {"phone": phone or "unknown", "updated_at": _now(),
            "conditions": list(conditions or [])}
    p = profile_path_for(phone)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def load_dsm5_profile(phone: str) -> dict:
    p = profile_path_for(phone)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Corrupt DSM-5 profile %s", p)
    return {"phone": phone or "unknown", "conditions": []}


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
             candidates, next_question, future_checkin, dsm5_criteria=None) -> None:
    """Record one triage step and update the patient's rolling profile.

    `dsm5_criteria` is a list of {condition, criterion, status}.
    `future_checkin` is a list of {days, message}.
    """
    session["turns"].append({
        "at": _now(),
        "descriptions": list(descriptions or []),
        "empathy": empathy or "",
        "candidates": list(candidates or []),
        "dsm5_criteria": list(dsm5_criteria or []),
        "next_question": next_question or "",
        "future_checkin": list(future_checkin or []),
    })
    # Rolling patient profile: each turn re-assesses against ALL symptoms, so the
    # latest differential + DSM-5 criteria IS the current profile.
    profile = record.setdefault("profile", {})
    profile["updated_at"] = _now()
    profile["candidates"] = list(candidates or [])
    profile["dsm5_criteria"] = list(dsm5_criteria or [])

    # Roll new future check-ins ({days, message}) into the patient-level pending
    # list, deduped by message, so run_checkin.py can ask them on a later call.
    pending = record.setdefault("pending_checkins", [])
    seen = {c["message"] for c in pending if isinstance(c, dict) and c.get("message")}
    for c in future_checkin or []:
        if isinstance(c, dict) and c.get("message") and c["message"] not in seen:
            # saved_at lets run_checkin.py honor the `days` schedule (filter_due).
            pending.append({"days": c.get("days"), "message": c["message"],
                            "saved_at": _now()})
            seen.add(c["message"])


def add_transcript(session: dict, role: str, text: str) -> None:
    if text:
        session.setdefault("transcript", []).append({"role": role, "text": text})


def pending_checkins(phone: str) -> list:
    return load(phone).get("pending_checkins", [])


def filter_due(pending: list, now=None) -> list:
    """Return the check-ins whose `days` have elapsed since `saved_at`.

    An item missing saved_at or days (or unparseable) is treated as due, so old
    records and schedule-less questions still get asked.
    """
    now = now or datetime.now(timezone.utc)
    due = []
    for c in pending or []:
        if not isinstance(c, dict):
            due.append(c)
            continue
        saved, days = c.get("saved_at"), c.get("days")
        if not saved or days is None:
            due.append(c)
            continue
        try:
            if now >= datetime.fromisoformat(saved) + timedelta(days=days):
                due.append(c)
        except Exception:
            due.append(c)
    return due


def remove_checkins(record: dict, items: list) -> None:
    """Drop the given check-ins (matched by message) from pending — call after a
    check-in call has actually asked them, so unasked/not-yet-due ones survive."""
    msgs = {c.get("message") for c in items if isinstance(c, dict)}
    record["pending_checkins"] = [
        c for c in record.get("pending_checkins", [])
        if not (isinstance(c, dict) and c.get("message") in msgs)
    ]


def clear_pending(record: dict) -> None:
    record["pending_checkins"] = []


def key_for(phone: str) -> str:
    """Public, filename-safe patient key (digits only) — what the dashboard and
    the /api/patients/{phone} route use to address a patient."""
    return _safe(phone)


# ---------------------------------------------------------------------------
# DAM acoustic scores — folded into the patient's longitudinal record so the
# depression/anxiety trend lives alongside the conversational differential.
# Called from analysis.analyze_call (a detached background task) AFTER the call
# is over, so it re-reads from disk, mutates, and writes back — no concurrent
# writer to this phone's file at that point.
# ---------------------------------------------------------------------------
def attach_scores(phone: str, call_id: str, scores: dict) -> None:
    """Attach a call's DAM scores to its session and append to the patient's
    rolling depression/anxiety score history (the longitudinal trend)."""
    record = load(phone)
    for s in record.get("sessions", []):
        if s.get("call_id") == call_id:
            s["scores"] = scores
            break
    profile = record.setdefault("profile", {})
    profile["latest_scores"] = scores
    history = profile.setdefault("scores_history", [])
    history.append({
        "call_id": call_id,
        "at": _now(),
        "status": scores.get("status", "ok"),
        "depression": scores.get("depression"),
        "depression_label": scores.get("depression_label"),
        "anxiety": scores.get("anxiety"),
        "anxiety_label": scores.get("anxiety_label"),
        "speaking_rate_wpm": scores.get("speaking_rate_wpm"),
        "indeterminate": scores.get("indeterminate"),
    })
    save(record)
    log.info("Attached DAM scores for call %s to patient %s", call_id, key_for(phone))


# ---------------------------------------------------------------------------
# Dashboard listing — one summary per patient for the nav column.
# ---------------------------------------------------------------------------
def _last_seen(record: dict) -> str:
    """Most recent activity timestamp: profile update, else newest session."""
    prof_ts = (record.get("profile") or {}).get("updated_at")
    sess_ts = [s.get("started_at") for s in record.get("sessions", []) if s.get("started_at")]
    return max([t for t in ([prof_ts] + sess_ts) if t], default="")


def list_summaries() -> list:
    """One lightweight summary per patient (phone), newest activity first."""
    if not DIR.exists():
        return []
    out = []
    for p in sorted(DIR.glob("*.json")):
        # Skip the sidecar DSM-5 profile and any atomic-write temp files.
        if p.name.endswith(".dsm5.json") or p.suffix == ".tmp":
            continue
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Corrupt interaction file %s; skipping in listing", p)
            continue
        profile = record.get("profile") or {}
        latest = profile.get("latest_scores") or {}
        out.append({
            "phone": record.get("phone") or p.stem,
            "key": p.stem,
            "last_seen": _last_seen(record),
            "sessions": len(record.get("sessions", [])),
            "candidates": list(profile.get("candidates") or []),
            "pending_checkins": len(record.get("pending_checkins", [])),
            "latest_scores": {
                "status": latest.get("status", "ok"),
                "depression": latest.get("depression"),
                "depression_label": latest.get("depression_label"),
                "anxiety": latest.get("anxiety"),
                "anxiety_label": latest.get("anxiety_label"),
                "indeterminate": latest.get("indeterminate"),
            } if latest else None,
        })
    out.sort(key=lambda r: r["last_seen"], reverse=True)
    return out
