from __future__ import annotations

import torch
from torch import nn

from csi300_multimodal.models.fusion import build_fusion


class SequenceBranchEncoder(nn.Module):
    def __init__(
        self,
        family: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_heads: int,
        max_len: int,
    ) -> None:
        super().__init__()
        self.family = family
        self.hidden_dim = hidden_dim
        if family == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif family == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.positional = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
                dim_feedforward=hidden_dim * 4,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unsupported deep model family: {family}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.family == "lstm":
            output, _ = self.encoder(features)
            return output[:, -1, :]
        encoded = self.input_proj(features) + self.positional[:, : features.size(1), :]
        output = self.encoder(encoded)
        return output[:, -1, :]


class DeepForecastModel(nn.Module):
    def __init__(self, config: dict, input_dims: dict[str, int]) -> None:
        super().__init__()
        family = config["model"]["family"]
        modality = config["model"]["modality"]
        hidden_dim = int(config["model"]["hidden_dim"])
        num_layers = int(config["model"]["num_layers"])
        dropout = float(config["model"]["dropout"])
        num_heads = int(config["model"]["num_heads"])
        max_len = int(config["features"]["lookback"])
        task = config["model"]["task"]

        self.modality = modality
        self.task = task

        self.num_encoder = None
        self.text_encoder = None
        if modality in {"num", "multimodal"}:
            self.num_encoder = SequenceBranchEncoder(
                family=family,
                input_dim=input_dims["numeric_dim"],
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                num_heads=num_heads,
                max_len=max_len,
            )
        if modality in {"text", "multimodal"}:
            self.text_encoder = SequenceBranchEncoder(
                family=family,
                input_dim=input_dims["text_dim"],
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                num_heads=num_heads,
                max_len=max_len,
            )

        if modality == "multimodal":
            self.fusion = build_fusion(config["model"]["fusion"], hidden_dim)
            head_dim = self.fusion.output_dim
        else:
            self.fusion = None
            head_dim = hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_head = nn.Linear(hidden_dim, 1) if task in {"cls", "multitask"} else None
        self.reg_head = nn.Linear(hidden_dim, 1) if task in {"reg", "multitask"} else None

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.modality == "num":
            return self.num_encoder(batch["x_num"])
        if self.modality == "text":
            return self.text_encoder(batch["x_text"])
        num_repr = self.num_encoder(batch["x_num"])
        text_repr = self.text_encoder(batch["x_text"])
        return self.fusion(num_repr, text_repr)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        shared = self.shared(self.encode(batch))
        outputs: dict[str, torch.Tensor] = {}
        if self.cls_head is not None:
            outputs["cls_logits"] = self.cls_head(shared).squeeze(-1)
        if self.reg_head is not None:
            outputs["regression"] = self.reg_head(shared).squeeze(-1)
        return outputs
