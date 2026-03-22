from __future__ import annotations

from csi300_multimodal.training.trainer import train_experiment


def run_train(config: dict):
    return train_experiment(config)
