from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from csi300_multimodal.data.contracts import (
    TEXT_COUNT_FEATURE_COLUMNS,
    TEXT_DAILY_FEATURE_COLUMNS,
    TEXT_LENGTH_FEATURE_COLUMNS,
    TEXT_SENTIMENT_FEATURE_COLUMNS,
)
from csi300_multimodal.text.aggregation import embedding_columns


def _build_trade_frame(trade_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    frame = pd.DataFrame({"trade_date": pd.to_datetime(list(trade_dates)).astype("datetime64[ns]")})
    return frame.sort_values("trade_date").reset_index(drop=True)


def build_daily_text_features(aligned_news: pd.DataFrame, trade_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    daily = _build_trade_frame(trade_dates)
    if aligned_news.empty:
        for column in TEXT_COUNT_FEATURE_COLUMNS + TEXT_LENGTH_FEATURE_COLUMNS:
            daily[column] = 0.0
        daily["news_count"] = daily["news_count"].astype(int)
        daily["has_news"] = daily["has_news"].astype(int)
        daily["finbert_pos_mean"] = 0.0
        daily["finbert_neg_mean"] = 0.0
        daily["finbert_neu_mean"] = 1.0
        daily["sentiment_score"] = 0.0
        daily["sentiment_abs"] = 0.0
        return daily[["trade_date"] + TEXT_DAILY_FEATURE_COLUMNS]

    frame = aligned_news.copy()
    title_series = frame["title_clean"] if "title_clean" in frame.columns else frame["title"]
    content_series = frame["content_clean"] if "content_clean" in frame.columns else frame["content"]
    frame["title_len"] = title_series.fillna("").str.len()
    frame["content_len"] = content_series.fillna("").str.len()
    grouped = frame.groupby("trade_date", as_index=False).agg(
        news_count=("news_id", "count"),
        title_len_mean=("title_len", "mean"),
        title_len_median=("title_len", "median"),
        title_len_min=("title_len", "min"),
        title_len_max=("title_len", "max"),
        content_len_mean=("content_len", "mean"),
        content_len_median=("content_len", "median"),
        content_len_min=("content_len", "min"),
        content_len_max=("content_len", "max"),
    )
    grouped["has_news"] = (grouped["news_count"] > 0).astype(int)
    grouped["finbert_pos_mean"] = 0.0
    grouped["finbert_neg_mean"] = 0.0
    grouped["finbert_neu_mean"] = 1.0
    grouped["sentiment_score"] = 0.0
    grouped["sentiment_abs"] = 0.0

    merged = daily.merge(grouped, on="trade_date", how="left")
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["has_news"] = merged["has_news"].fillna(0).astype(int)
    for column in TEXT_LENGTH_FEATURE_COLUMNS:
        merged[column] = merged[column].fillna(0.0)
    for column in TEXT_SENTIMENT_FEATURE_COLUMNS:
        default = 1.0 if column == "finbert_neu_mean" else 0.0
        merged[column] = merged[column].fillna(default)
    return merged[["trade_date"] + TEXT_DAILY_FEATURE_COLUMNS]


def build_placeholder_daily_embeddings(
    trade_dates: Iterable[pd.Timestamp],
    embedding_dim: int,
) -> pd.DataFrame:
    daily = _build_trade_frame(trade_dates)
    columns = embedding_columns(embedding_dim)
    zeros = pd.DataFrame(np.zeros((len(daily), embedding_dim), dtype=float), columns=columns)
    return pd.concat([daily, zeros], axis=1)
