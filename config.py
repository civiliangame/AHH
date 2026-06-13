"""Central config loaded from environment (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name} (see .env.example)")
    return val


XAI_API_KEY = _require("XAI_API_KEY")
XAI_REALTIME_URL = os.getenv("XAI_REALTIME_URL", "wss://api.x.ai/v1/realtime")
# grok-voice-latest -> newest; or pin e.g. grok-voice-think-fast-1.0
XAI_REALTIME_MODEL = os.getenv("XAI_REALTIME_MODEL", "grok-voice-latest")
# Voice: eve, ara, rex, sal, leo, or a custom voice ID.
XAI_VOICE = os.getenv("XAI_VOICE", "ara")
XAI_INSTRUCTIONS = os.getenv(
    "XAI_INSTRUCTIONS",
    "You are Grok, a helpful voice assistant speaking to a caller in real time "
    "over the phone. Keep replies short, natural, and conversational since they "
    "are spoken aloud.",
)
# The line Grok speaks first when the call connects (spoken by the realtime
# model in its own voice). Empty = wait for the caller to talk first.
XAI_GREETING = os.getenv(
    "XAI_GREETING",
    "Hi, I'm Pulse, your AI assistant. How can I help you today?",
)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Telnyx (Call Control / Voice API) ---
# API key (starts with "KEY...") used to answer calls and start streaming.
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY", "")
TELNYX_API_BASE = os.getenv("TELNYX_API_BASE", "https://api.telnyx.com/v2")
# Public host Telnyx should open the media WebSocket to. Usually your ngrok
# domain (no scheme), e.g. "abc123.ngrok-free.app". If blank, we derive it
# from the inbound webhook's Host header.
PUBLIC_HOSTNAME = os.getenv("PUBLIC_HOSTNAME", "")
# Path the Telnyx media stream connects to.
STREAM_PATH = os.getenv("STREAM_PATH", "/media-stream")

# Telnyx Media Streaming uses G.711 μ-law (PCMU) @ 8 kHz, mono. xAI supports
# "audio/pcmu" natively, so audio passes through with zero transcoding.
XAI_AUDIO_FORMAT = "audio/pcmu"
