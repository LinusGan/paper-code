from __future__ import annotations

import pickle
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from csi300_multimodal.config import resolve_path
from csi300_multimodal.models.datasets import (
    SequenceDataset,
    build_tabular_dataset,
    merge_embeddings,
    resolve_feature_spaces,
)
from csi300_multimodal.models.registry import build_model
from csi300_multimodal.training.scalers import FrameStandardizer
from csi300_multimodal.utils.io import dump_json, dump_yaml, ensure_dir, read_table, write_table
from csi300_multimodal.utils.seed import seed_everything


@dataclass
class TrainingResult:
    run_dir: Path
    predictions_path: Path
    metrics_path: Path
    model_path: Path


def build_run_name(config: dict) -> str:
    explicit = config["data"].get("run_name")
    if explicit:
        return str(explicit)
    model_cfg = config["model"]
    return "_".join(
        [
            str(model_cfg["family"]),
            str(model_cfg["modality"]),
            str(model_cfg["task"]),
            str(model_cfg["text_repr"]),
            str(model_cfg.get("fusion", "none")),
        ]
    )


def build_run_dir(config: dict) -> Path:
    output_dir = Path(config["data"]["output_dir"])
    run_dir = output_dir / "runs" / build_run_name(config)
    ensure_dir(run_dir)
    return run_dir


def load_training_frames(config: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    main_frame = read_table(resolve_path(config, "main_table_path"))
    main_frame["trade_date"] = pd.to_datetime(main_frame["trade_date"]).dt.normalize()
    embeddings = None
    embeddings_path = resolve_path(config, "daily_embeddings_path")
    if embeddings_path.exists():
        embeddings = read_table(embeddings_path)
        embeddings["trade_date"] = pd.to_datetime(embeddings["trade_date"]).dt.normalize()
    return main_frame, embeddings


def select_scaler_columns(feature_spaces: dict[str, list[str]]) -> list[str]:
    return list(feature_spaces["numeric_columns"])


def compute_batch_loss(config: dict, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    import torch
    from torch import nn

    task = config["model"]["task"]
    weights = config["train"]["task_weights"]
    total_loss = torch.tensor(0.0, device=batch["y_reg"].device)
    stats: dict[str, float] = {}

    if task in {"cls", "multitask"}:
        cls_loss = nn.BCEWithLogitsLoss()(outputs["cls_logits"], batch["y_cls"])
        total_loss = total_loss + float(weights.get("cls", 1.0)) * cls_loss
        stats["cls_loss"] = float(cls_loss.detach().cpu())
    if task in {"reg", "multitask"}:
        reg_loss = nn.MSELoss()(outputs["regression"], batch["y_reg"])
        total_loss = total_loss + float(weights.get("reg", 1.0)) * reg_loss
        stats["reg_loss"] = float(reg_loss.detach().cpu())

    stats["loss"] = float(total_loss.detach().cpu())
    return total_loss, stats


def move_batch_to_device(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    import torch

    moved: dict[str, torch.Tensor | list[str]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(
    model,
    loader,
    config: dict,
    optimizer,
    device,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals: dict[str, float] = {}
    count = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        if train_mode:
            optimizer.zero_grad()
        outputs = model(batch)
        loss, stats = compute_batch_loss(config, outputs, batch)
        if train_mode:
            loss.backward()
            optimizer.step()
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1

    if count == 0:
        return {"loss": float("nan")}
    return {key: value / count for key, value in totals.items()}


def predict_deep_model(
    model,
    loader,
    config: dict,
    device,
) -> pd.DataFrame:
    import torch

    model.eval()
    records: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(batch)
            task = config["model"]["task"]
            cls_scores = None
            cls_labels = None
            pred_reg = None
            if task in {"cls", "multitask"}:
                cls_scores = torch.sigmoid(outputs["cls_logits"]).detach().cpu().numpy()
                cls_labels = (cls_scores >= float(config["eval"]["threshold"])).astype(int)
            if task in {"reg", "multitask"}:
                pred_reg = outputs["regression"].detach().cpu().numpy()

            trade_dates = list(raw_batch["trade_date"])
            for index, trade_date in enumerate(trade_dates):
                record = {
                    "trade_date": trade_date,
                    "y_cls": float(raw_batch["y_cls"][index]),
                    "next_ret_1": float(raw_batch["y_reg"][index]),
                    "next_oc_return": float(raw_batch["next_oc_return"][index]),
                    "trend_regime": int(raw_batch["trend_regime"][index]),
                    "vol_regime": int(raw_batch["vol_regime"][index]),
                    "drawdown_state": int(raw_batch["drawdown_state"][index]),
                }
                if cls_scores is not None:
                    record["pred_cls_score"] = float(cls_scores[index])
                    record["pred_cls_label"] = int(cls_labels[index])
                else:
                    record["pred_cls_score"] = np.nan
                    record["pred_cls_label"] = np.nan
                if pred_reg is not None:
                    record["pred_reg"] = float(pred_reg[index])
                else:
                    record["pred_reg"] = np.nan
                records.append(record)
    return pd.DataFrame.from_records(records)


def train_deep_model(frame: pd.DataFrame, config: dict, run_dir: Path) -> TrainingResult:
    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    feature_spaces = resolve_feature_spaces(frame, config)
    scaler = FrameStandardizer(columns=select_scaler_columns(feature_spaces))
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    scaler.fit(train_frame)
    scaled_frame = scaler.transform(frame)

    lookback = int(config["features"]["lookback"])
    modality = str(config["model"]["modality"])
    train_dataset = SequenceDataset(
        frame=scaled_frame,
        split="train",
        lookback=lookback,
        numeric_columns=feature_spaces["numeric_columns"] if modality in {"num", "multimodal"} else [],
        text_columns=feature_spaces["text_columns"] if modality in {"text", "multimodal"} else [],
        modality=modality,
    )
    valid_dataset = SequenceDataset(
        frame=scaled_frame,
        split="valid",
        lookback=lookback,
        numeric_columns=feature_spaces["numeric_columns"] if modality in {"num", "multimodal"} else [],
        text_columns=feature_spaces["text_columns"] if modality in {"text", "multimodal"} else [],
        modality=modality,
    )
    test_dataset = SequenceDataset(
        frame=scaled_frame,
        split="test",
        lookback=lookback,
        numeric_columns=feature_spaces["numeric_columns"] if modality in {"num", "multimodal"} else [],
        text_columns=feature_spaces["text_columns"] if modality in {"text", "multimodal"} else [],
        modality=modality,
    )

    input_dims = {
        "numeric_dim": len(feature_spaces["numeric_columns"]) if modality in {"num", "multimodal"} else 0,
        "text_dim": len(feature_spaces["text_columns"]) if modality in {"text", "multimodal"} else 0,
    }
    model = build_model(config, input_dims=input_dims)
    device = torch.device(config["text"].get("device", "cpu"))
    model.to(device)

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if len(train_dataset) == 0:
        raise RuntimeError("No train sequences were produced. Check split dates and lookback.")

    optimizer = Adam(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = int(config["train"]["early_stopping_patience"])
    epochs = int(config["train"]["epochs"])
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(model, train_loader, config, optimizer, device)
        valid_stats = run_epoch(model, valid_loader, config, None, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats.get("loss", float("nan")),
                "valid_loss": valid_stats.get("loss", float("nan")),
            }
        )
        current_loss = valid_stats.get("loss", float("inf"))
        if not isfinite(current_loss):
            current_loss = train_stats.get("loss", float("inf"))
        if current_loss < best_loss or best_state is None:
            best_loss = current_loss
            best_state = model.state_dict()
            best_epoch = epoch
        if epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("Deep model training produced no checkpoint.")

    model_path = run_dir / "model.pt"
    torch.save(best_state, model_path)
    model.load_state_dict(best_state)

    predictions = []
    for split, loader in [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:
        pred_frame = predict_deep_model(model, loader, config, device)
        pred_frame["split"] = split
        predictions.append(pred_frame)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame["trade_date"] = pd.to_datetime(prediction_frame["trade_date"])

    predictions_path = run_dir / "predictions.parquet"
    metrics_path = run_dir / "training_summary.json"
    write_table(prediction_frame, predictions_path)
    dump_json(
        {
            "best_epoch": best_epoch,
            "best_valid_loss": best_loss,
            "history": history,
            "scaler": scaler.to_dict(),
        },
        metrics_path,
    )
    return TrainingResult(run_dir=run_dir, predictions_path=predictions_path, metrics_path=metrics_path, model_path=model_path)


def train_tabular_model(frame: pd.DataFrame, config: dict, run_dir: Path) -> TrainingResult:
    feature_spaces = resolve_feature_spaces(frame, config)
    scaler = FrameStandardizer(columns=select_scaler_columns(feature_spaces))
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    scaler.fit(train_frame)
    scaled_frame = scaler.transform(frame)
    dataset_train = build_tabular_dataset(scaled_frame, "train", feature_spaces["input_columns"])
    model = build_model(config, input_dims={"numeric_dim": len(feature_spaces["input_columns"]), "text_dim": 0})
    model.fit(dataset_train.X, dataset_train.y_cls, dataset_train.y_reg)

    prediction_frames: list[pd.DataFrame] = []
    for split in ["train", "valid", "test"]:
        dataset = build_tabular_dataset(scaled_frame, split, feature_spaces["input_columns"])
        split_predictions = model.predict(dataset.X)
        pred_frame = dataset.metadata
        pred_frame["pred_cls_score"] = split_predictions.get("pred_cls_score", np.nan)
        pred_frame["pred_cls_label"] = split_predictions.get("pred_cls_label", np.nan)
        pred_frame["pred_reg"] = split_predictions.get("pred_reg", np.nan)
        prediction_frames.append(pred_frame)

    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    model_path = run_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    predictions_path = run_dir / "predictions.parquet"
    metrics_path = run_dir / "training_summary.json"
    write_table(prediction_frame, predictions_path)
    dump_json({"scaler": scaler.to_dict(), "backend": config["model"]["family"]}, metrics_path)
    return TrainingResult(run_dir=run_dir, predictions_path=predictions_path, metrics_path=metrics_path, model_path=model_path)


def train_experiment(config: dict) -> TrainingResult:
    seed_everything(int(config["train"]["seed"]))
    run_dir = build_run_dir(config)
    dump_yaml(config, run_dir / "config.yaml")

    main_frame, embeddings = load_training_frames(config)
    merged = merge_embeddings(main_frame, embeddings if config["model"]["text_repr"] == "cls_embedding" else None)
    family = config["model"]["family"]
    if family == "xgboost":
        return train_tabular_model(merged, config, run_dir)
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"torch import failed in this environment: {probe.stderr.strip()}")
    return train_deep_model(merged, config, run_dir)
