from __future__ import annotations

from pathlib import Path

import pandas as pd

from csi300_multimodal.data.contracts import NEWS_ARTICLE_COLUMNS, RAW_MARKET_COLUMNS, RAW_NEWS_COLUMNS
from csi300_multimodal.data.schemas import NewsArticle
from csi300_multimodal.utils.io import read_table


def validate_columns(frame: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def load_market_raw(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    validate_columns(frame, RAW_MARKET_COLUMNS, "market")
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    return frame


def load_news_raw(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    validate_columns(frame, RAW_NEWS_COLUMNS, "news")
    frame = frame.copy().reset_index(drop=True)
    frame["publish_time"] = pd.to_datetime(frame["publish_time"], utc=False, errors="coerce")
    frame["title"] = frame["title"].fillna("")
    frame["content"] = frame["content"].fillna("")
    if "source" not in frame.columns:
        frame["source"] = "legacy_input"
    if "channel" not in frame.columns:
        frame["channel"] = ""
    if "crawl_time" not in frame.columns:
        frame["crawl_time"] = frame["publish_time"]
    else:
        frame["crawl_time"] = pd.to_datetime(frame["crawl_time"], utc=False, errors="coerce").fillna(frame["publish_time"])

    articles = [NewsArticle.from_record(record).to_record() for record in frame.to_dict(orient="records")]
    normalized = pd.DataFrame.from_records(articles, columns=NEWS_ARTICLE_COLUMNS)
    extra_columns = [column for column in frame.columns if column not in normalized.columns]
    for column in extra_columns:
        normalized[column] = frame[column].values
    return normalized.sort_values("publish_time").reset_index(drop=True)
