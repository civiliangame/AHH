"""In-process event hub + JSONL persistence for live transcript display.

Each call's turns are appended to transcripts/<call_id>.jsonl AND broadcast to
any connected dashboards over SSE (see GET /events in server.py).

Single-process only — perfect for the demo. If you ever run multiple uvicorn
workers, swap this hub for Redis pub/sub so events fan out across processes.
"""
import asyncio
import json
import time
from collections import deque
from pathlib import Path

TRANSCRIPT_DIR = Path("transcripts")
TRANSCRIPT_DIR.mkdir(exist_ok=True)


class EventHub:
    def __init__(self, history: int = 500):
        self._subs: set[asyncio.Queue] = set()
        self._recent = deque(maxlen=history)  # replayed to late-joining dashboards

    async def publish(self, event: dict, remember: bool = True):
        if remember:
            self._recent.append(event)
        for q in list(self._subs):
            q.put_nowait(event)

    async def subscribe(self):
        q: asyncio.Queue = asyncio.Queue()
        for e in self._recent:          # so a dashboard opened mid-call isn't blank
            q.put_nowait(e)
        self._subs.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._subs.discard(q)


hub = EventHub()


def _persist(call_id: str, role: str, text: str, ts: float):
    line = json.dumps({"call_id": call_id, "role": role, "text": text, "ts": ts})
    with open(TRANSCRIPT_DIR / f"{call_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")


async def emit(call_id: str, role: str, text: str, *, partial: bool = False):
    """Broadcast one transcript event. Finals are persisted + remembered for
    replay; partials (live agent deltas) are streamed only, never stored."""
    if not text:
        return
    ev = {"call_id": call_id, "role": role, "text": text,
          "ts": time.time(), "partial": partial}
    await hub.publish(ev, remember=not partial)
    if not partial:
        _persist(call_id, role, text, ev["ts"])


async def emit_metadata(call_id: str, data: dict):
    """Broadcast a structured triage-metadata event (symptom, differential, ...).
    Persisted to the call's JSONL so a refresh replays it."""
    ev = {"call_id": call_id, "role": "metadata", "data": data,
          "ts": time.time(), "partial": False}
    await hub.publish(ev, remember=True)
    line = json.dumps(ev)
    with open(TRANSCRIPT_DIR / f"{call_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")
