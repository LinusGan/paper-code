from __future__ import annotations

import logging

import pandas as pd

from csi300_multimodal.config import resolve_path
from csi300_multimodal.data.calendar import build_trade_calendar
from csi300_multimodal.data.io import load_market_raw, load_news_raw
from csi300_multimodal.data.schemas import NewsArticle
from csi300_multimodal.preprocess.align_trade_date import assign_trade_dates
from csi300_multimodal.preprocess.build_daily_text_features import (
    build_daily_text_features,
    build_placeholder_daily_embeddings,
)
from csi300_multimodal.preprocess.clean_news import clean_news_frame
from csi300_multimodal.preprocess.deduplicate import deduplicate_news_frame
from csi300_multimodal.utils.io import read_table, write_table

LOGGER = logging.getLogger(__name__)


def run_prepare_news(config: dict) -> None:
    prepared_market_path = resolve_path(config, "prepared_market_path")
    if prepared_market_path.exists():
        market = read_table(prepared_market_path)
    else:
        market = load_market_raw(resolve_path(config, "market_path"))
    trade_dates = build_trade_calendar(market)

    raw_news_path = resolve_path(config, "raw_news_path")
    legacy_news_path = resolve_path(config, "news_path")
    if raw_news_path.exists():
        input_path = raw_news_path
    elif legacy_news_path.exists():
        input_path = legacy_news_path
    else:
        LOGGER.warning("No raw news input found at %s or %s. Writing empty processed artifacts.", raw_news_path, legacy_news_path)
        news = pd.DataFrame(columns=NewsArticle.columns())
    if raw_news_path.exists() or legacy_news_path.exists():
        news = load_news_raw(input_path)

    cleaned = clean_news_frame(news, title_fallback=bool(config["text"].get("title_fallback", True)))
    deduped, unique_news = deduplicate_news_frame(
        cleaned,
        title_similarity_threshold=float(config["news"]["title_similarity_threshold"]),
        dedup_window_hours=int(config["news"]["dedup_window_hours"]),
    )
    aligned = assign_trade_dates(
        news_frame=unique_news,
        trade_dates=trade_dates,
        timezone=str(config["news"]["timezone"]),
        close_time=str(config["news"]["market_close_time"]),
    )
    daily_features = build_daily_text_features(aligned_news=aligned, trade_dates=trade_dates)
    daily_embeddings = build_placeholder_daily_embeddings(
        trade_dates=trade_dates,
        embedding_dim=int(config["text"]["embedding_dim"]),
    )

    write_table(cleaned, resolve_path(config, "cleaned_news_path"))
    write_table(deduped, resolve_path(config, "deduped_news_path"))
    write_table(aligned, resolve_path(config, "aligned_news_path"))
    write_table(daily_features, resolve_path(config, "daily_text_features_path"))
    write_table(daily_embeddings, resolve_path(config, "daily_embeddings_path"))
