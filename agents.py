"""Voice agent personas.

Each named persona is a distinct agent: its own system prompt, voice, opening
line, and tools. The media-stream WebSocket picks one by the `?agent=` query
param on the stream URL:

  - inbound calls  -> default "triage" (no param)        : your DSM-5 triage bot
  - outbound calls -> "checkin" (run_checkin.py sets it) : a wellness check-in bot

Add a persona here, point a stream URL at `?agent=<name>`, and it's live.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import config
import triage


@dataclass
class Persona:
    name: str
    instructions: str      # system prompt
    voice: str             # eve, ara, rex, sal, leo, or a custom voice id
    greeting: str          # spoken first when the call connects
    tools: list = field(default_factory=list)            # xAI tool schemas
    handler: Optional[Callable[[str, dict], dict]] = None  # runs a tool call


def _read(path: str, fallback: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return fallback


# Inbound triage agent — existing behavior, from config / system_prompt.txt.
TRIAGE = Persona(
    name="triage",
    instructions=config.XAI_INSTRUCTIONS,
    voice=config.XAI_VOICE,
    greeting=config.XAI_GREETING,
    tools=triage.TRIAGE_TOOLS,
    handler=triage.handle,
)

# Outbound check-in agent — dialed by run_checkin.py. Edit checkin_prompt.txt
# to change its persona/script.
CHECKIN = Persona(
    name="checkin",
    instructions=_read(
        "checkin_prompt.txt",
        "You are Eva from AtLegionX making a brief, warm outbound wellness "
        "check-in call to a patient you have spoken with before. You are speaking "
        "aloud over the phone. Confirm it's a good time, ask how they've been "
        "since you last talked, listen, and keep it short and caring.",
    ),
    voice=config.CHECKIN_VOICE,
    greeting=config.CHECKIN_GREETING,
    tools=[],        # no tools for the check-in agent (add here if needed)
    handler=None,
)

_REGISTRY = {p.name: p for p in (TRIAGE, CHECKIN)}


def get(name: Optional[str]) -> Persona:
    """Return the persona by name, defaulting to triage (inbound)."""
    return _REGISTRY.get(name or "triage", TRIAGE)
