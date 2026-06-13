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
XAI_REALTIME_MODEL = os.getenv("XAI_REALTIME_MODEL", "grok-realtime")
XAI_VOICE = os.getenv("XAI_VOICE", "ara")
XAI_INSTRUCTIONS = os.getenv(
    "XAI_INSTRUCTIONS",
    "You are a friendly, concise voice assistant answering a phone call. "
    "Keep replies short and natural for speech.",
)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Telnyx Media Streaming sends G.711 μ-law (PCMU) at 8 kHz, mono.
# We ask xAI to use the same so audio passes through without transcoding.
TELNYX_AUDIO_FORMAT = "g711_ulaw"
TELNYX_SAMPLE_RATE = 8000
