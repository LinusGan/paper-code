from __future__ import annotations

RAW_MARKET_COLUMNS = ["trade_date", "open", "high", "low", "close", "volume"]
RAW_NEWS_COLUMNS = ["publish_time", "title", "content"]
NEWS_ARTICLE_COLUMNS = [
    "news_id",
    "source",
    "url",
    "title",
    "publish_time",
    "content",
    "crawl_time",
    "channel",
    "raw_title",
    "raw_content",
    "title_clean",
    "content_clean",
    "title_hash",
    "content_hash",
    "is_duplicate",
    "duplicate_reason",
    "canonical_news_id",
    "trade_date",
]
BASE_MARKET_INPUT_COLUMNS = ["open", "high", "low", "close", "volume"]

RETURN_COLUMNS = ["ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20"]
VOLATILITY_COLUMNS = ["volatility_5", "volatility_10", "volatility_20"]
PRICE_SHAPE_COLUMNS = ["hl_range", "oc_return", "gap_return"]
MA_COLUMNS = ["ma_5", "ma_10", "ma_20", "ma_gap_5", "ma_gap_10", "ma_gap_20"]
MOMENTUM_COLUMNS = ["rsi_14", "momentum_5", "momentum_10"]
VOLUME_COLUMNS = ["volume_change_1", "volume_change_5", "price_volume_corr_5", "price_volume_corr_10"]
REGIME_COLUMNS = ["trend_regime", "vol_regime", "drawdown_state"]

MARKET_FEATURE_COLUMNS = (
    RETURN_COLUMNS
    + VOLATILITY_COLUMNS
    + PRICE_SHAPE_COLUMNS
    + MA_COLUMNS
    + MOMENTUM_COLUMNS
    + VOLUME_COLUMNS
    + REGIME_COLUMNS
)

TEXT_COUNT_FEATURE_COLUMNS = [
    "news_count",
    "has_news",
]

TEXT_LENGTH_FEATURE_COLUMNS = [
    "title_len_mean",
    "title_len_median",
    "title_len_min",
    "title_len_max",
    "content_len_mean",
    "content_len_median",
    "content_len_min",
    "content_len_max",
]

TEXT_SENTIMENT_FEATURE_COLUMNS = [
    "finbert_pos_mean",
    "finbert_neg_mean",
    "finbert_neu_mean",
    "sentiment_score",
    "sentiment_abs",
]

TEXT_SENTIMENT_DAILY_COLUMNS = TEXT_COUNT_FEATURE_COLUMNS + TEXT_SENTIMENT_FEATURE_COLUMNS
TEXT_DAILY_FEATURE_COLUMNS = TEXT_COUNT_FEATURE_COLUMNS + TEXT_LENGTH_FEATURE_COLUMNS + TEXT_SENTIMENT_FEATURE_COLUMNS

LABEL_COLUMNS = ["next_ret_1", "next_oc_return", "y_cls", "split"]
NON_INPUT_COLUMNS = ["trade_date"] + LABEL_COLUMNS

DEFAULT_SPLITS = {
    "train_end": "2022-12-31",
    "valid_end": "2023-12-31",
    "test_end": "2024-12-31",
}
