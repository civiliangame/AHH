from typing import Any, Mapping, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperConfig, WhisperModel


class WhisperEncoderBackbone(torch.nn.Module):
    def __init__(
        self,
        model: str = "openai/whisper-small.en",
        hf_config: Optional[Mapping[str, Any]] = None,
        lora_params: Optional[Mapping[str, Any]] = None,
    ):
        """Whisper encoder model with optional Low-Rank Adaptation.

        Parameters
        ----------
        model: Name of WhisperModel whose encoder to load from HuggingFace
        hf_config: Optional config for HuggingFace model
        lora_params: Parameters for Low-Rank Adaptation

        """
        super().__init__()
        hf_config = hf_config if hf_config is not None else dict()
        backbone_config = WhisperConfig.from_pretrained(model, **hf_config)
        self.backbone = (
            WhisperModel.from_pretrained(
                model,
                config=backbone_config,
            )
            .get_encoder()
            .train()
        )
        if lora_params is not None and len(lora_params) > 0:
            lora_config = LoraConfig(**lora_params)
            self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone_dim = backbone_config.hidden_size

    def forward(self, whisper_feature_batch):
        return self.backbone(whisper_feature_batch).last_hidden_state.mean(dim=1)


class SharedLayers(torch.nn.Module):
    def __init__(self, input_dim: int, proj_dims: list[int]):
        """Fully connected network with Mish nonlinearities between linear layers. No nonlinearity at input or output.

        Parameters
        ----------
        input_dim: Dimension of input features
        proj_dims: Dimensions of layers to create

        """
        super().__init__()
        modules = []
        for output_dim in proj_dims[:-1]:
            modules.extend([torch.nn.Linear(input_dim, output_dim), torch.nn.Mish()])
            input_dim = output_dim
        modules.append(torch.nn.Linear(input_dim, proj_dims[-1]))
        self.shared_layers = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.shared_layers(x)

class TaskHead(torch.nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, dropout: float = 0.0):
        """Fully connected network with one hidden layer, dropout, and a scalar output."""
        super().__init__()

        self.linear = torch.nn.Linear(input_dim, proj_dim)
        self.activation = torch.nn.Mish()
        self.dropout = torch.nn.Dropout(dropout)
        self.final_layer = torch.nn.Linear(proj_dim, 1, bias=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.final_layer(x)
        return x


class MultitaskHead(torch.nn.Module):
    def __init__(
        self,
        backbone_dim: int,
        shared_projection_dim: list[int],
        tasks: Mapping[str, Mapping[str, Any]],
    ):
        """Fully connected network with multiple named scalar outputs."""
        super().__init__()

        # Initialize the shared network and task-specific networks
        self.shared_layers = SharedLayers(backbone_dim, shared_projection_dim)
        self.classifier_head = torch.nn.ModuleDict(
            {
                task: TaskHead(shared_projection_dim[-1], **task_config)
                for task, task_config in tasks.items()
            }
        )

    def forward(self, x):
        x = self.shared_layers(x)
        return {task: head(x) for task, head in self.classifier_head.items()}


def average_tensor_in_segments(tensor: torch.Tensor, lengths: list[int] | torch.Tensor):
    """Average segments of a `tensor` along dimension 0 based on a list of `lengths`

    For example, with input tensor `t` and `lengths` [1, 3, 2], the output would be
    [t[0], (t[1] + t[2] + t[3]) / 3, (t[4] + t[5]) / 2]

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to average
    lengths : list of ints
        The lengths of each segment to average in the tensor, in order

    Returns
    -------
    torch.Tensor
        The tensor with relevant segments averaged
    """
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, device=tensor.device)
    index = torch.repeat_interleave(
        torch.arange(len(lengths), device=tensor.device), lengths
    )
    out = torch.zeros(
        lengths.shape + tensor.shape[1:], device=tensor.device, dtype=tensor.dtype
    )
    out.index_add_(0, index, tensor)
    broadcastable_lengths = lengths.view((-1,) + (1,) * (len(out.shape) - 1))
    return out / broadcastable_lengths


class Classifier(torch.nn.Module):
    def __init__(
        self,
        backbone_configs: Mapping[str, Mapping[str, Any]],
        classifier_config: Mapping[str, Any],
        inference_thresholds: Mapping[str, Any],
        preprocessor_config: Mapping[str, Any],
    ):
        """Full Kintsugi Depression and Anxiety model.

        Whisper encoder -> Mean pooling over time -> Layers shared across tasks -> Per-task heads

        Parameters
        ----------
        backbone_configs:
        classifier_config:
        inference_thresholds:
        preprocessor_config:

        """
        super().__init__()

        self.backbone = torch.nn.ModuleDict(
            {
                key: WhisperEncoderBackbone(**backbone_configs[key])
                for key in sorted(backbone_configs.keys())
            }
        )

        backbone_dim = sum(layer.backbone_dim for layer in self.backbone.values())
        self.head = MultitaskHead(backbone_dim, **classifier_config)
        self.inference_thresholds = inference_thresholds
        self.preprocessor_config = preprocessor_config

    def forward(self, x, lengths):
        backbone_outputs = {
            key: average_tensor_in_segments(layer(x), lengths)
            for key, layer in self.backbone.items()
        }
        backbone_output = torch.cat(list(backbone_outputs.values()), dim=1)
        return self.head(backbone_output), torch.ones_like(lengths)

    def quantize_scores(self, scores: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Map per-task scores to discrete predictions per `inference_thresholds` config."""
        return {
            key: torch.searchsorted(torch.tensor(self.inference_thresholds[key], device=value.device), value.mean(), out_int32=True)
            for key, value in scores.items()
        }
