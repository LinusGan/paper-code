from __future__ import annotations

from csi300_multimodal.config import resolve_path
from csi300_multimodal.data.calendar import build_trade_calendar
from csi300_multimodal.text.aggregation import aggregate_embeddings_daily, aggregate_sentiment_daily
from csi300_multimodal.preprocess.build_daily_text_features import (
    build_daily_text_features,
    build_placeholder_daily_embeddings,
)
from csi300_multimodal.utils.io import read_table, write_table


def run_build_dataset(config: dict) -> None:
    market = read_table(resolve_path(config, "prepared_market_path"))
    market["trade_date"] = market["trade_date"].astype("datetime64[ns]")
    trade_dates = build_trade_calendar(market)
    daily_features_path = resolve_path(config, "daily_text_features_path")
    if daily_features_path.exists():
        daily_features = read_table(daily_features_path)
        daily_features["trade_date"] = daily_features["trade_date"].astype("datetime64[ns]")
    else:
        daily_features = build_daily_text_features(aligned_news=market.iloc[0:0].copy(), trade_dates=trade_dates)

    encoded_path = resolve_path(config, "encoded_news_path")
    if encoded_path.exists():
        encoded_news = read_table(encoded_path)
        encoded_news["trade_date"] = encoded_news["trade_date"].astype("datetime64[ns]")
        sentiment_daily = aggregate_sentiment_daily(encoded_news=encoded_news, trade_dates=trade_dates)
        sentiment_columns = [column for column in sentiment_daily.columns if column != "trade_date"]
        daily_features = daily_features.drop(columns=[column for column in sentiment_columns if column in daily_features.columns])
        daily_features = daily_features.merge(sentiment_daily, on="trade_date", how="left")
        embedding_daily = aggregate_embeddings_daily(
            encoded_news=encoded_news,
            trade_dates=trade_dates,
            embedding_dim=int(config["text"]["embedding_dim"]),
        )
    else:
        encoded_news = market.iloc[0:0].copy()
        embedding_path = resolve_path(config, "daily_embeddings_path")
        if embedding_path.exists():
            embedding_daily = read_table(embedding_path)
            embedding_daily["trade_date"] = embedding_daily["trade_date"].astype("datetime64[ns]")
        else:
            embedding_daily = build_placeholder_daily_embeddings(
                trade_dates=trade_dates,
                embedding_dim=int(config["text"]["embedding_dim"]),
            )

    main_table = market.merge(daily_features, on="trade_date", how="left")
    write_table(daily_features, daily_features_path)
    write_table(main_table, resolve_path(config, "main_table_path"))
    write_table(embedding_daily, resolve_path(config, "daily_embeddings_path"))
