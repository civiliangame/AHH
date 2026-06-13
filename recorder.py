"""Per-call caller-audio recorder.

Buffers inbound μ-law @ 8 kHz frames (Base64-encoded by Telnyx), and on save()
decodes + resamples to 16 kHz PCM16 mono WAV via the existing audio helper.
That format is what dam/pipeline.py expects.
"""
import base64
import wave
from pathlib import Path

from audio import ulaw8k_to_pcm16

RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

TARGET_RATE = 16000  # DAM expects 16 kHz mono


class CallRecorder:
    def __init__(self, call_id: str):
        self.call_id = call_id
        self._chunks: list[bytes] = []

    def feed(self, payload_b64: str) -> None:
        """Append one Telnyx media frame's worth of μ-law bytes."""
        if not payload_b64:
            return
        try:
            self._chunks.append(base64.b64decode(payload_b64))
        except Exception:
            # Bad frame — skip silently; phone calls drop frames all the time.
            pass

    @property
    def duration_seconds(self) -> float:
        # μ-law @ 8 kHz: 1 byte per sample.
        return sum(len(c) for c in self._chunks) / 8000.0

    def save(self) -> Path | None:
        """Write recordings/{call_id}.wav. Returns None if nothing was captured."""
        if not self._chunks:
            return None
        ulaw = b"".join(self._chunks)
        pcm16 = ulaw8k_to_pcm16(ulaw, target_rate=TARGET_RATE)
        path = RECORDINGS_DIR / f"{self.call_id}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # PCM16
            wf.setframerate(TARGET_RATE)
            wf.writeframes(pcm16)
        return path
