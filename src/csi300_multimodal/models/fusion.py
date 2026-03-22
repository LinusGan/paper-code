from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.output_dim = hidden_dim * 2

    def forward(self, num_repr: torch.Tensor, text_repr: torch.Tensor) -> torch.Tensor:
        return torch.cat([num_repr, text_repr], dim=-1)


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_dim = hidden_dim

    def compute_gate(self, num_repr: torch.Tensor, text_repr: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(torch.cat([num_repr, text_repr], dim=-1)))

    def forward(self, num_repr: torch.Tensor, text_repr: torch.Tensor) -> torch.Tensor:
        gate = self.compute_gate(num_repr, text_repr)
        return gate * num_repr + (1.0 - gate) * text_repr


def build_fusion(name: str, hidden_dim: int) -> nn.Module:
    if name == "concat":
        return ConcatFusion(hidden_dim)
    if name == "gate":
        return GatedFusion(hidden_dim)
    raise ValueError(f"Unsupported fusion: {name}")
