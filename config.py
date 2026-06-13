"""Central config loaded from environment (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name} (see .env.example)")
    return val

def get_system_prompt():
    with open("system_prompt.txt", "r") as f:
        return f.read()



XAI_API_KEY = _require("XAI_API_KEY")
XAI_REALTIME_URL = os.getenv("XAI_REALTIME_URL", "wss://api.x.ai/v1/realtime")
# grok-voice-latest -> newest; or pin e.g. grok-voice-think-fast-1.0
XAI_REALTIME_MODEL = os.getenv("XAI_REALTIME_MODEL", "grok-voice-think-fast-1.1")
REASONING = ""
TURN_DETECTION_threshold = ""
VAD = ""

# --- Grok text model for the triage differential step (get_next_question) ---
XAI_API_BASE = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
# Grok chat model for the differential step. grok-4.3, run with reasoning
# disabled (reasoning.effort = "none" in the request) for lower latency.
GROK_TRIAGE_MODEL = os.getenv("GROK_TRIAGE_MODEL", "grok-4.3")
# Collection (vector store) of DSM-5 reference material used to ground candidates.
DSM5_COLLECTION_ID = os.getenv(
    "DSM5_COLLECTION_ID", "collection_44862946-181e-4ce3-b710-bb6f3df35f07")


# Voice: eve, ara, rex, sal, leo, or a custom voice ID.
XAI_VOICE = os.getenv("XAI_VOICE", "ara")
XAI_INSTRUCTIONS = get_system_prompt()
# The line Grok speaks first when the call connects (spoken by the realtime
# model in its own voice). Empty = wait for the caller to talk first.
XAI_GREETING = "Hi, I'm Eva from AtLegionX. How are you feeling today?"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Outbound check-in persona (separate agent; see agents.py) ---
CHECKIN_VOICE = os.getenv("CHECKIN_VOICE", "rex")
CHECKIN_GREETING = os.getenv(
    "CHECKIN_GREETING",
    "Hi, this is Eva from AtLegionX calling to check in. "
    "Is now a good time to talk for a couple of minutes?",
)

# --- Telnyx (Call Control / Voice API) ---
# API key (starts with "KEY...") used to answer calls and start streaming.
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY", "")
TELNYX_API_BASE = os.getenv("TELNYX_API_BASE", "https://api.telnyx.com/v2")
# For OUTBOUND calls (run_checkin.py): the Call Control App's connection id and
# a Telnyx number on it to call FROM. Find both in the Telnyx portal.
TELNYX_CONNECTION_ID = os.getenv("TELNYX_CONNECTION_ID", "")
TELNYX_FROM_NUMBER = os.getenv("TELNYX_FROM_NUMBER", "")
# Public host Telnyx should open the media WebSocket to. Usually your ngrok
# domain (no scheme), e.g. "abc123.ngrok-free.app". If blank, we derive it
# from the inbound webhook's Host header.
PUBLIC_HOSTNAME = os.getenv("PUBLIC_HOSTNAME", "").strip()
# Accept a full URL or a bare host; we only want host[:port] for the wss:// URL.
if "://" in PUBLIC_HOSTNAME:
    PUBLIC_HOSTNAME = PUBLIC_HOSTNAME.split("://", 1)[1]
PUBLIC_HOSTNAME = PUBLIC_HOSTNAME.strip("/")
# Path the Telnyx media stream connects to.
STREAM_PATH = os.getenv("STREAM_PATH", "/media-stream")

# Telnyx Media Streaming uses G.711 μ-law (PCMU) @ 8 kHz, mono. xAI supports
# "audio/pcmu" natively, so audio passes through with zero transcoding.
XAI_AUDIO_FORMAT = "audio/pcmu"
