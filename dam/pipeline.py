import os
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Union
import torch

from .config import default_config
from .featex import load_audio, Preprocessor
from .model import Classifier

class Pipeline:
    def __init__(self, checkpoint: Optional[str | Path] = None, config: Optional[Mapping[str, Any]] = None, device: Optional[torch.device] = None):
        if checkpoint is None:
            file_dir = Path(__file__).parent.resolve()
            checkpoint = file_dir / "dam3.1.ckpt"
        if config is None:
            config = default_config
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model = Classifier(**config)
        self.preprocessor = Preprocessor(**self.model.preprocessor_config)
        state_dict = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def run_on_features(self, features: torch.Tensor, quantize: bool = True):
        scores = self.model(features, torch.tensor([features.shape[0]], device=self.device))[0]
        if quantize:
            return {k: int(v.item()) for k, v in self.model.quantize_scores(scores).items()}
        else:
            return scores

    def run_on_audio(self, audio: torch.Tensor, quantize: bool = True):
        features = self.preprocessor.preprocess_with_audio_normalization(audio)
        return self.run_on_features(features.to(self.device), quantize=quantize)

    def run_on_file(self, source: Union[BinaryIO, str, os.PathLike], quantize=True):
        audio = load_audio(source)
        return self.run_on_audio(audio, quantize=quantize)
