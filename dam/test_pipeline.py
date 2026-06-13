"""Basic smoke test for the DAM pipeline.

Usage:
    python test_pipeline.py                 # uses sample.wav, or generates one
    python test_pipeline.py path/to/clip.m4a  # auto-converts via ffmpeg

Accepts WAV/FLAC/OGG directly (soundfile backend). Anything else
(.m4a, .mp3, etc.) is transcoded to a temp 16kHz mono WAV with ffmpeg.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pipeline import Pipeline

SAMPLE_PATH = Path(__file__).parent / "sample.wav"
SAMPLE_RATE = 16000
DURATION_S = 30
NATIVE_EXTS = {".wav", ".flac", ".ogg"}


def ensure_sample_wav() -> Path:
    if SAMPLE_PATH.exists():
        print(f"Using existing sample: {SAMPLE_PATH}")
        return SAMPLE_PATH

    print(f"Generating synthetic {DURATION_S}s mono 16kHz WAV at {SAMPLE_PATH}")
    rng = np.random.default_rng(0)
    n = SAMPLE_RATE * DURATION_S
    t = np.arange(n) / SAMPLE_RATE
    audio = (
        0.3 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t)
        + 0.05 * rng.standard_normal(n)
    ).astype(np.float32)
    sf.write(SAMPLE_PATH, audio, SAMPLE_RATE, subtype="PCM_16")
    return SAMPLE_PATH


def transcode_to_wav(src: Path) -> Path:
    if not shutil.which("ffmpeg"):
        sys.exit(f"ffmpeg not on PATH — install it (brew install ffmpeg) to decode {src.suffix}")
    dst = Path(tempfile.mkstemp(suffix=".wav", prefix="dam_")[1])
    print(f"Transcoding {src.name} -> {dst} (16 kHz mono)")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ar", str(SAMPLE_RATE), "-ac", "1", str(dst)],
        check=True,
    )
    return dst


def resolve_input(argv: list[str]) -> Path:
    if len(argv) > 1:
        src = Path(argv[1]).expanduser().resolve()
        if not src.exists():
            sys.exit(f"No such file: {src}")
        return src if src.suffix.lower() in NATIVE_EXTS else transcode_to_wav(src)
    return ensure_sample_wav()


def main() -> None:
    wav = resolve_input(sys.argv)

    print("Loading pipeline (this downloads whisper-small.en on first run)...")
    pipeline = Pipeline()
    print(f"Device: {pipeline.device}")

    print("\nRaw scores:")
    raw = pipeline.run_on_file(wav, quantize=False)
    if isinstance(raw, torch.Tensor):
        print(raw)
    else:
        print({k: float(v) for k, v in raw.items()})

    print("\nQuantized scores:")
    print(pipeline.run_on_file(wav, quantize=True))


if __name__ == "__main__":
    main()
