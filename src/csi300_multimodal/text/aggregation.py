from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from csi300_multimodal.data.contracts import TEXT_SENTIMENT_DAILY_COLUMNS


def _build_trade_frame(trade_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    frame = pd.DataFrame({"trade_date": pd.to_datetime(list(trade_dates)).astype("datetime64[ns]")})
    return frame.sort_values("trade_date").reset_index(drop=True)


def aggregate_sentiment_daily(encoded_news: pd.DataFrame, trade_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    daily = _build_trade_frame(trade_dates)
    if encoded_news.empty:
        daily["news_count"] = 0
        daily["has_news"] = 0
        daily["finbert_pos_mean"] = 0.0
        daily["finbert_neg_mean"] = 0.0
        daily["finbert_neu_mean"] = 1.0
        daily["sentiment_score"] = 0.0
        daily["sentiment_abs"] = 0.0
        return daily[["trade_date"] + TEXT_SENTIMENT_DAILY_COLUMNS]

    grouped = encoded_news.groupby("trade_date", as_index=False).agg(
        news_count=("news_id", "count"),
        finbert_pos_mean=("pos", "mean"),
        finbert_neg_mean=("neg", "mean"),
        finbert_neu_mean=("neu", "mean"),
    )
    grouped["has_news"] = (grouped["news_count"] > 0).astype(int)
    grouped["sentiment_score"] = grouped["finbert_pos_mean"] - grouped["finbert_neg_mean"]
    grouped["sentiment_abs"] = grouped["sentiment_score"].abs()

    merged = daily.merge(grouped, on="trade_date", how="left")
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["has_news"] = merged["has_news"].fillna(0).astype(int)
    merged["finbert_pos_mean"] = merged["finbert_pos_mean"].fillna(0.0)
    merged["finbert_neg_mean"] = merged["finbert_neg_mean"].fillna(0.0)
    merged["finbert_neu_mean"] = merged["finbert_neu_mean"].fillna(1.0)
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
    merged["sentiment_abs"] = merged["sentiment_abs"].fillna(0.0)
    return merged[["trade_date"] + TEXT_SENTIMENT_DAILY_COLUMNS]


def embedding_columns(embedding_dim: int) -> list[str]:
    return [f"emb_{index:03d}" for index in range(embedding_dim)]


def aggregate_embeddings_daily(
    encoded_news: pd.DataFrame,
    trade_dates: Iterable[pd.Timestamp],
    embedding_dim: int,
) -> pd.DataFrame:
    daily = _build_trade_frame(trade_dates)
    columns = embedding_columns(embedding_dim)
    if encoded_news.empty:
        zeros = pd.DataFrame(np.zeros((len(daily), embedding_dim), dtype=float), columns=columns)
        return pd.concat([daily, zeros], axis=1)

    group_frames: list[pd.DataFrame] = []
    for trade_date, group in encoded_news.groupby("trade_date"):
        values = group[columns].to_numpy(dtype=float)
        pooled = values.mean(axis=0) if len(values) else np.zeros(embedding_dim, dtype=float)
        payload = {"trade_date": pd.Timestamp(trade_date)}
        payload.update({name: pooled[idx] for idx, name in enumerate(columns)})
        group_frames.append(pd.DataFrame([payload]))

    grouped = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame(columns=["trade_date"] + columns)
    merged = daily.merge(grouped, on="trade_date", how="left").fillna(0.0)
    return merged[["trade_date"] + columns]
