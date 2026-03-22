from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from csi300_multimodal.data.contracts import (
    BASE_MARKET_INPUT_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    REGIME_COLUMNS,
    TEXT_DAILY_FEATURE_COLUMNS,
)


def merge_embeddings(main_frame: pd.DataFrame, embeddings_frame: pd.DataFrame | None) -> pd.DataFrame:
    if embeddings_frame is None:
        return main_frame.copy()
    merged = main_frame.merge(embeddings_frame, on="trade_date", how="left")
    embedding_cols = [column for column in merged.columns if column.startswith("emb_")]
    if embedding_cols:
        merged[embedding_cols] = merged[embedding_cols].fillna(0.0)
    return merged


def resolve_feature_spaces(frame: pd.DataFrame, config: dict) -> dict[str, list[str]]:
    numeric_columns = BASE_MARKET_INPUT_COLUMNS + MARKET_FEATURE_COLUMNS
    if config["features"].get("exclude_regime_inputs", True):
        numeric_columns = [column for column in numeric_columns if column not in REGIME_COLUMNS]

    text_repr = config["model"]["text_repr"]
    if text_repr == "sentiment":
        text_columns = list(TEXT_DAILY_FEATURE_COLUMNS)
    elif text_repr == "cls_embedding":
        text_columns = sorted(column for column in frame.columns if column.startswith("emb_"))
    else:
        raise ValueError(f"Unsupported text representation: {text_repr}")

    modality = config["model"]["modality"]
    if modality == "num":
        input_columns = list(numeric_columns)
    elif modality == "text":
        input_columns = list(text_columns)
    elif modality == "multimodal":
        input_columns = list(numeric_columns) + list(text_columns)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return {
        "numeric_columns": numeric_columns,
        "text_columns": text_columns,
        "input_columns": input_columns,
    }


@dataclass
class TabularDataset:
    frame: pd.DataFrame
    feature_columns: list[str]

    @property
    def X(self) -> np.ndarray:
        return self.frame[self.feature_columns].to_numpy(dtype=float)

    @property
    def y_cls(self) -> np.ndarray:
        return self.frame["y_cls"].to_numpy(dtype=float)

    @property
    def y_reg(self) -> np.ndarray:
        return self.frame["next_ret_1"].to_numpy(dtype=float)

    @property
    def metadata(self) -> pd.DataFrame:
        keep = [
            "trade_date",
            "split",
            "y_cls",
            "next_ret_1",
            "next_oc_return",
            "trend_regime",
            "vol_regime",
            "drawdown_state",
        ]
        return self.frame[keep].copy()


def build_tabular_dataset(frame: pd.DataFrame, split: str, feature_columns: list[str]) -> TabularDataset:
    subset = frame.loc[
        (frame["split"] == split) & frame["next_ret_1"].notna() & frame[feature_columns].notna().all(axis=1)
    ].reset_index(drop=True)
    return TabularDataset(frame=subset, feature_columns=feature_columns)


class SequenceDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        split: str,
        lookback: int,
        numeric_columns: list[str],
        text_columns: list[str],
        modality: str,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.split = split
        self.lookback = lookback
        self.numeric_columns = numeric_columns
        self.text_columns = text_columns
        self.modality = modality
        self.indices = self._build_indices()

    def _build_indices(self) -> list[int]:
        indices: list[int] = []
        for index in range(self.lookback - 1, len(self.frame)):
            window = self.frame.iloc[index - self.lookback + 1 : index + 1]
            row = self.frame.iloc[index]
            if row["split"] != self.split or pd.isna(row["next_ret_1"]):
                continue
            if not (window["split"] == self.split).all():
                continue
            indices.append(index)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, np.ndarray | float | int | str]:
        end_index = self.indices[item]
        window = self.frame.iloc[end_index - self.lookback + 1 : end_index + 1]
        row = self.frame.iloc[end_index]
        x_num = window[self.numeric_columns].to_numpy(dtype=np.float32) if self.numeric_columns else np.zeros((self.lookback, 0), dtype=np.float32)
        x_text = window[self.text_columns].to_numpy(dtype=np.float32) if self.text_columns else np.zeros((self.lookback, 0), dtype=np.float32)
        return {
            "trade_date": pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d"),
            "x_num": x_num,
            "x_text": x_text,
            "y_cls": float(row["y_cls"]),
            "y_reg": float(row["next_ret_1"]),
            "next_oc_return": float(row["next_oc_return"]),
            "trend_regime": float(row["trend_regime"]),
            "vol_regime": float(row["vol_regime"]),
            "drawdown_state": float(row["drawdown_state"]),
        }
