from __future__ import annotations

from csi300_multimodal.features.market import compute_market_features
from csi300_multimodal.models.datasets import SequenceDataset
from csi300_multimodal.text.aggregation import aggregate_sentiment_daily

from tests.conftest import make_market_frame


def test_sequence_dataset_respects_split_boundaries() -> None:
    market = make_market_frame(periods=30)
    prepared = compute_market_features(
        market_frame=market,
        split_dates={"train_end": "2022-10-25", "valid_end": "2022-11-08", "test_end": "2022-12-31"},
        fillna_value=0.0,
    )
    sentiment = aggregate_sentiment_daily(encoded_news=prepared.iloc[0:0].copy(), trade_dates=prepared["trade_date"])
    merged = prepared.merge(sentiment, on="trade_date", how="left")

    dataset = SequenceDataset(
        frame=merged,
        split="valid",
        lookback=3,
        numeric_columns=["open", "close", "ret_1"],
        text_columns=["news_count", "sentiment_score"],
        modality="multimodal",
    )

    assert len(dataset) > 0
    first = dataset[0]
    assert first["trade_date"] >= "2022-10-28"
    assert first["x_num"].shape == (3, 3)
    assert first["x_text"].shape == (3, 2)
