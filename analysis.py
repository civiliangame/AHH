"""DAM (depression/anxiety model) post-call analysis.

Loads the DAM pipeline (Whisper-Small + classification heads) once at server
startup, then scores each call's caller audio after the call ends. Results are
persisted to scores/{call_id}.json and broadcast on the live SSE stream as a
metadata event so the dashboard renders them inline with the transcript.

In addition to the headline depression/anxiety bands, we surface:
  - raw model logits (depression_raw, anxiety_raw)
  - per-chunk severity arc (each ~30 s window, before mean-pooling) — lets the
    dashboard plot the call's affect trajectory
  - margin to the nearest band threshold + an `indeterminate` flag for calls
    that sit borderline (margin < DAM_INDETERMINATE_MARGIN)
  - speech_seconds (energy-VAD on the recording) + caller_word_count (from the
    persisted transcript) → speaking_rate_wpm, a well-replicated voice biomarker
"""
import asyncio
import bisect
import json
import logging
import os
from pathlib import Path

import events

_DAM_DIR = Path(__file__).parent / "dam"

SCORES_DIR = Path("scores")
SCORES_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR = Path("transcripts")

DEFAULT_CHECKPOINT = _DAM_DIR / "dam3.1.ckpt"
DAM_CHECKPOINT = Path(os.getenv("DAM_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
DAM_MIN_SECONDS = float(os.getenv("DAM_MIN_SECONDS", "5"))
# Calls with raw score within this distance of a band threshold are flagged
# "indeterminate" — clinical-decision support to avoid over-trusting borderline
# classifications. The depression/anxiety threshold gaps are ~0.38, so 0.05 is
# the inner ~13 % of each band.
DAM_INDETERMINATE_MARGIN = float(os.getenv("DAM_INDETERMINATE_MARGIN", "0.05"))

# Band labels — see dam/README.md (PHQ-9 / GAD-7 thresholds).
DEPRESSION_LABELS = ["none", "mild-moderate", "severe"]
ANXIETY_LABELS = ["none", "mild", "moderate", "severe"]
LABELS = {"depression": DEPRESSION_LABELS, "anxiety": ANXIETY_LABELS}

log = logging.getLogger("analysis")

_pipeline = None


def init_pipeline() -> None:
    """Load the ~702 MB DAM checkpoint once at server startup."""
    global _pipeline
    if _pipeline is not None:
        return
    from dam.pipeline import Pipeline  # vendored under ./dam (package)
    log.info("Loading DAM pipeline checkpoint %s …", DAM_CHECKPOINT)
    _pipeline = Pipeline(checkpoint=DAM_CHECKPOINT)
    log.info("DAM pipeline ready (device=%s)", _pipeline.device)


def _band_and_margin(raw: float, thresholds: list[float]) -> tuple[int, float]:
    """Bucket `raw` into a band index per `thresholds` (matches DAM's searchsorted)
    and return the distance from `raw` to the nearest threshold (the 'margin')."""
    band = bisect.bisect_left(thresholds, raw)
    margin = min(abs(raw - t) for t in thresholds)
    return band, margin


def _score_sync(wav_path: Path) -> dict:
    """One forward pass: returns pooled band+raw, per-chunk arc, margins."""
    import torch
    from dam.featex import load_audio

    if _pipeline is None:
        raise RuntimeError("DAM pipeline not initialized; call init_pipeline() first")

    audio = load_audio(str(wav_path))
    features = _pipeline.preprocessor.preprocess_with_audio_normalization(audio)
    features = features.to(_pipeline.device)
    # `features` shape: (num_chunks, 80 mel bands, 3000 frames) per dam/featex.py.

    with torch.no_grad():
        # Per-chunk Whisper encoder outputs — each chunk already mean-pooled over
        # its 30 s window inside WhisperEncoderBackbone.forward.
        per_chunk_features = {
            key: layer(features) for key, layer in _pipeline.model.backbone.items()
        }
        per_chunk_concat = torch.cat(list(per_chunk_features.values()), dim=1)  # (N, 1536)
        # Two head passes:
        #   1) `pooled` — head applied to the mean-pooled features (matches the
        #      trained classifier path; this is the authoritative score).
        #   2) `per_chunk` — head applied to each chunk's features independently;
        #      an approximation used purely for the temporal arc visualization.
        pooled_features = per_chunk_concat.mean(dim=0, keepdim=True)  # (1, 1536)
        pooled_logits = _pipeline.model.head(pooled_features)
        per_chunk_logits = _pipeline.model.head(per_chunk_concat)

    out: dict = {"per_chunk": []}
    n_chunks = int(per_chunk_concat.shape[0])
    indeterminate = False

    for task in ("depression", "anxiety"):
        raw_val = float(pooled_logits[task].squeeze().item())
        thresholds = list(_pipeline.model.inference_thresholds[task])
        band, margin = _band_and_margin(raw_val, thresholds)
        labels = LABELS[task]
        out[task] = band
        out[f"{task}_raw"] = raw_val
        out[f"{task}_label"] = labels[min(band, len(labels) - 1)]
        out[f"{task}_margin"] = margin
        if margin < DAM_INDETERMINATE_MARGIN:
            indeterminate = True

    out["indeterminate"] = indeterminate
    out["indeterminate_margin"] = DAM_INDETERMINATE_MARGIN
    out["n_chunks"] = n_chunks

    # Per-chunk arc: one entry per ~30 s window, with raw score + band per task.
    dep_thresh = list(_pipeline.model.inference_thresholds["depression"])
    anx_thresh = list(_pipeline.model.inference_thresholds["anxiety"])
    for i in range(n_chunks):
        dep_raw = float(per_chunk_logits["depression"][i].squeeze().item())
        anx_raw = float(per_chunk_logits["anxiety"][i].squeeze().item())
        out["per_chunk"].append({
            "depression_raw": dep_raw,
            "depression": bisect.bisect_left(dep_thresh, dep_raw),
            "anxiety_raw": anx_raw,
            "anxiety": bisect.bisect_left(anx_thresh, anx_raw),
        })

    return out


def _speech_seconds(wav_path: Path) -> float:
    """Rough energy-based VAD: fraction of 20 ms frames whose RMS exceeds a
    floor-or-peak-relative threshold. Caller-only audio, so no need to
    distinguish speakers — just speech vs. silence."""
    import torch
    import torchaudio

    audio, sr = torchaudio.load(str(wav_path))
    if audio.numel() == 0:
        return 0.0
    audio = audio.mean(dim=0)  # downmix to mono (already mono, but safe)
    frame_ms = 20
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = audio.shape[0] // frame_len
    if n_frames == 0:
        return 0.0
    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = frames.pow(2).mean(dim=1).sqrt()
    peak = float(rms.max().item())
    threshold = max(0.005, peak * 0.06)  # noise floor or 6 % of peak
    speech_frames = int((rms > threshold).sum().item())
    return speech_frames * frame_ms / 1000.0


def _caller_word_count(call_id: str) -> int:
    """Count caller-role words in transcripts/{call_id}.jsonl. The xAI realtime
    model produces user transcripts during the call (see server.py:235); we
    persist them via events._persist, so this is just a re-read."""
    path = TRANSCRIPT_DIR / f"{call_id}.jsonl"
    if not path.exists():
        return 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("role") == "caller":
            total += len(str(ev.get("text", "")).split())
    return total


def _voice_metrics(call_id: str, wav_path: Path) -> dict:
    """Speaking-rate biomarker: WPM over actively-spoken time."""
    speech = _speech_seconds(wav_path)
    words = _caller_word_count(call_id)
    wpm = (words / speech * 60.0) if speech > 0 else 0.0
    return {
        "speech_seconds": round(speech, 2),
        "caller_word_count": words,
        "speaking_rate_wpm": round(wpm, 1),
    }


def _write_atomic(call_id: str, payload: dict) -> Path:
    path = SCORES_DIR / f"{call_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return path


async def analyze_call(call_id: str, wav_path: Path, duration_seconds: float) -> None:
    """Background task: score the WAV, persist, broadcast a 'scores' metadata event."""
    payload: dict = {"call_id": call_id, "duration_seconds": round(duration_seconds, 2)}
    try:
        if duration_seconds < DAM_MIN_SECONDS:
            payload["status"] = "skipped"
            payload["reason"] = f"audio too short ({duration_seconds:.1f}s < {DAM_MIN_SECONDS}s)"
            _write_atomic(call_id, payload)
            log.info("DAM skipped %s: %s", call_id, payload["reason"])
            return

        scores = await asyncio.to_thread(_score_sync, wav_path)
        voice = await asyncio.to_thread(_voice_metrics, call_id, wav_path)

        payload["status"] = "ok"
        payload.update(scores)
        payload.update(voice)
        _write_atomic(call_id, payload)
        log.info(
            "DAM scored %s: depression=%d (%s, margin=%.3f) anxiety=%d (%s, margin=%.3f)"
            " indet=%s wpm=%.1f chunks=%d",
            call_id,
            scores["depression"], scores["depression_label"], scores["depression_margin"],
            scores["anxiety"], scores["anxiety_label"], scores["anxiety_margin"],
            scores["indeterminate"], voice["speaking_rate_wpm"], scores["n_chunks"],
        )
        await events.emit_metadata(call_id, {"kind": "scores", **scores, **voice})
    except Exception as e:
        log.exception("DAM analysis failed for %s", call_id)
        payload["status"] = "error"
        payload["error"] = str(e)
        _write_atomic(call_id, payload)
