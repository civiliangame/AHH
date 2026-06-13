"""Optional audio transcoding helpers.

The default bridge configures xAI to speak G.711 μ-law @ 8 kHz, matching
Telnyx, so no transcoding happens. If xAI requires PCM16 (e.g. 24 kHz),
use these to convert in server.py.

Uses the stdlib `audioop` module (present in CPython <= 3.12; removed in 3.13).
On 3.13+, install `audioop-lts` or swap in numpy-based resampling.
"""
import audioop


def ulaw8k_to_pcm16(ulaw_bytes: bytes, target_rate: int = 24000) -> bytes:
    """G.711 μ-law @ 8 kHz  ->  linear PCM16 @ target_rate."""
    pcm16 = audioop.ulaw2lin(ulaw_bytes, 2)  # 2 = 16-bit samples
    if target_rate != 8000:
        pcm16, _ = audioop.ratecv(pcm16, 2, 1, 8000, target_rate, None)
    return pcm16


def pcm16_to_ulaw8k(pcm16_bytes: bytes, source_rate: int = 24000) -> bytes:
    """Linear PCM16 @ source_rate  ->  G.711 μ-law @ 8 kHz."""
    if source_rate != 8000:
        pcm16_bytes, _ = audioop.ratecv(pcm16_bytes, 2, 1, source_rate, 8000, None)
    return audioop.lin2ulaw(pcm16_bytes, 2)
