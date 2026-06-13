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
# Spoken first when the call connects (bot greets). Empty = wait for caller.
XAI_GREETING = os.getenv("XAI_GREETING", "Say hello and introduce yourself briefly.")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Telnyx Media Streaming uses G.711 μ-law (PCMU) @ 8 kHz, mono. xAI supports
# "audio/pcmu" natively, so audio passes through with zero transcoding.
XAI_AUDIO_FORMAT = "audio/pcmu"
