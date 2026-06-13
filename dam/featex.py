"""Preprocessing and normalization to prepare audio for Kintsugi Depression and Anxiety model."""
from typing import Union, BinaryIO
import numpy as np
import os
import torch
import torchaudio
from transformers import AutoFeatureExtractor

from .config import EXPECTED_SAMPLE_RATE, logmel_energies


def load_audio(source: Union[BinaryIO, str, os.PathLike]) -> torch.Tensor:
    """Load audio file, verify mono channel count, and resample if necessary.

    Parameters
    ----------
    source: open file or path to file

    Returns
    -------
    Time domain audio samples as a 1 x num_samples float tensor sampled at 16 kHz.

    """
    audio, fs = torchaudio.load(source)
    if audio.shape[0] != 1:
        raise ValueError(f"Provided audio has {audio.shape[0]} != 1 channels.")
    if fs != EXPECTED_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, fs, EXPECTED_SAMPLE_RATE)
    return audio


class Preprocessor:
    def __init__(self,
                 normalize_features: bool = True,
                 chunk_seconds: int = 30,
                 max_overlap_frac: float = 0.0,
                 pad_last_chunk_to_full: bool = True,
    ):
        """Create preprocessor object.

        Parameters
        ----------
        normalize_features: Whether the Whisper preprocessor should normalize features
        chunk_seconds: Size of model's receptive field in seconds
        max_overlap_frac: Fraction of each chunk allowed to overlap previous chunk for inputs longer than chunk_seconds
        pad_last_chunk_to_full: Whether to pad audio to an integer multiple of chunk_seconds

        """
        self.preprocessor = AutoFeatureExtractor.from_pretrained("openai/whisper-small.en")
        self.normalize_features = normalize_features
        self.chunk_seconds = chunk_seconds
        self.max_overlap_frac = max_overlap_frac
        self.pad_last_chunk_to_full = pad_last_chunk_to_full

    def preprocess_with_audio_normalization(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Run Whisper preprocessor and normalization expected by the model.

        Note: some normalization steps can be avoided, but are included to match
        feature extraction used during training.

        Parameters
        ----------
        audio: Raw audio samples as a 1 x num_samples float tensor sampled at 16 kHz

        Returns
        -------
        Normalized mel filter bank features as a float tensor of shape
        num_chunks x 80 mel filter bands x 3000 time frames

        """
        # Remove DC offset and scale amplitude to [-1, 1]
        audio = torch.squeeze(audio, 0)
        audio = audio - torch.mean(audio)
        audio = audio / torch.max(torch.abs(audio))

        chunk_samples = EXPECTED_SAMPLE_RATE * self.chunk_seconds

        if self.pad_last_chunk_to_full:
            # pad audio so that the last chunk is not dropped
            if self.max_overlap_frac > 0:
                raise ValueError(
                    f"pad_last_chunk_to_full is only supported for non-overlapping windows"
                )
            num_chunks = np.ceil(len(audio) / chunk_samples)
            pad_size = int(num_chunks * chunk_samples - len(audio))
            audio = torch.nn.functional.pad(audio, (0, pad_size))

        overflow_len = len(audio) - chunk_samples

        min_hop_samples = int(
            (1 - self.max_overlap_frac) * chunk_samples
        )

        n_windows = 1 + overflow_len // min_hop_samples
        window_starts = np.linspace(0, overflow_len, max(n_windows, 1)).astype(int)

        features = self.preprocessor(
            [
                audio[start : start + chunk_samples].numpy(force=True)
                for start in window_starts
            ],
            return_tensors="pt",
            sampling_rate=EXPECTED_SAMPLE_RATE,
            do_normalize=self.normalize_features,
        )
        for key in ("input_features", "input_values"):
            if hasattr(features, key):
                features = getattr(features, key)
                break

        mean_features = torch.mean(features, dim=-1)
        # features are [batch, n_logmel_bins, n_frames]
        rescale_factor = logmel_energies.unsqueeze(0) - mean_features
        rescale_factor = rescale_factor.unsqueeze(2)
        features += rescale_factor
        return features
