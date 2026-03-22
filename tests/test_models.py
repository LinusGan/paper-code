from __future__ import annotations

import pytest

from tests.conftest import torch_importable


def test_gated_fusion_gate_stays_in_unit_interval() -> None:
    if not torch_importable():
        pytest.skip("torch is not importable in this environment")
    import torch

    from csi300_multimodal.models.fusion import GatedFusion

    fusion = GatedFusion(hidden_dim=4)
    num_repr = torch.randn(2, 4)
    text_repr = torch.randn(2, 4)
    gate = fusion.compute_gate(num_repr, text_repr)
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)


def test_deep_model_multitask_shapes() -> None:
    if not torch_importable():
        pytest.skip("torch is not importable in this environment")
    import torch

    from csi300_multimodal.models.deep import DeepForecastModel

    config = {
        "model": {
            "family": "transformer",
            "modality": "multimodal",
            "task": "multitask",
            "fusion": "concat",
            "hidden_dim": 8,
            "dropout": 0.1,
            "num_layers": 1,
            "num_heads": 2,
        },
        "features": {"lookback": 5},
    }
    model = DeepForecastModel(config=config, input_dims={"numeric_dim": 6, "text_dim": 4})
    batch = {
        "x_num": torch.randn(3, 5, 6),
        "x_text": torch.randn(3, 5, 4),
    }
    outputs = model(batch)

    assert outputs["cls_logits"].shape == (3,)
    assert outputs["regression"].shape == (3,)
