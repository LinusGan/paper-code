from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _maybe_float(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    y_true_int = y_true.astype(int)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "accuracy": _maybe_float(accuracy_score(y_true_int, y_pred)),
        "precision": _maybe_float(precision_score(y_true_int, y_pred, zero_division=0)),
        "recall": _maybe_float(recall_score(y_true_int, y_pred, zero_division=0)),
        "f1": _maybe_float(f1_score(y_true_int, y_pred, zero_division=0)),
        "balanced_accuracy": _maybe_float(balanced_accuracy_score(y_true_int, y_pred)),
        "mcc": _maybe_float(matthews_corrcoef(y_true_int, y_pred)),
    }
    metrics["auc"] = _maybe_float(roc_auc_score(y_true_int, y_score)) if len(np.unique(y_true_int)) > 1 else None
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"mse": None, "rmse": None, "mae": None, "ic": None, "rank_ic": None}
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    pearson = frame["y_true"].corr(frame["y_pred"], method="pearson")
    spearman = frame["y_true"].corr(frame["y_pred"], method="spearman")
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": _maybe_float(mse),
        "rmse": _maybe_float(np.sqrt(mse)),
        "mae": _maybe_float(mean_absolute_error(y_true, y_pred)),
        "ic": _maybe_float(pearson),
        "rank_ic": _maybe_float(spearman),
    }


def evaluate_predictions(frame: pd.DataFrame, threshold: float) -> dict[str, dict[str, dict[str, float | None]]]:
    results: dict[str, dict[str, dict[str, float | None]]] = {}
    for split, group in frame.groupby("split"):
        split_metrics: dict[str, dict[str, float | None]] = {}
        if group["pred_cls_score"].notna().any():
            split_metrics["classification"] = classification_metrics(
                y_true=group["y_cls"].to_numpy(dtype=float),
                y_score=group["pred_cls_score"].to_numpy(dtype=float),
                threshold=threshold,
            )
        if group["pred_reg"].notna().any():
            split_metrics["regression"] = regression_metrics(
                y_true=group["next_ret_1"].to_numpy(dtype=float),
                y_pred=group["pred_reg"].to_numpy(dtype=float),
            )
        results[str(split)] = split_metrics
    return results


def evaluate_groups(frame: pd.DataFrame, threshold: float, group_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, float | int | str | None]] = []
    for split, split_frame in frame.groupby("split"):
        for column in group_columns:
            for value, group in split_frame.groupby(column):
                record: dict[str, float | int | str | None] = {"split": split, "group_column": column, "group_value": int(value)}
                if group["pred_cls_score"].notna().any():
                    cls = classification_metrics(
                        y_true=group["y_cls"].to_numpy(dtype=float),
                        y_score=group["pred_cls_score"].to_numpy(dtype=float),
                        threshold=threshold,
                    )
                    record.update({f"cls_{key}": value for key, value in cls.items()})
                if group["pred_reg"].notna().any():
                    reg = regression_metrics(
                        y_true=group["next_ret_1"].to_numpy(dtype=float),
                        y_pred=group["pred_reg"].to_numpy(dtype=float),
                    )
                    record.update({f"reg_{key}": value for key, value in reg.items()})
                records.append(record)
    return pd.DataFrame.from_records(records)
