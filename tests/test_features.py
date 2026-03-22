from __future__ import annotations

from csi300_multimodal.features.market import compute_market_features

from tests.conftest import make_market_frame


def test_market_features_are_generated_and_filled() -> None:
    market = make_market_frame(periods=40)
    features = compute_market_features(
        market_frame=market,
        split_dates={"train_end": "2022-11-01", "valid_end": "2022-11-30", "test_end": "2023-01-31"},
        fillna_value=0.0,
    )

    assert {"ret_1", "volatility_20", "ma_gap_20", "rsi_14", "trend_regime"} <= set(features.columns)
    assert features[["ret_1", "volatility_20", "ma_gap_20", "rsi_14"]].isna().sum().sum() == 0
    assert features["y_cls"].isna().iloc[-1]
