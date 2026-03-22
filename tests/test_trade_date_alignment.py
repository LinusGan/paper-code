from __future__ import annotations

import pandas as pd

from csi300_multimodal.preprocess.align_trade_date import assign_trade_dates, map_to_trade_date


def test_trade_date_alignment_respects_close_boundary_and_non_trading_days() -> None:
    trade_dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]))

    assert map_to_trade_date(pd.Timestamp("2024-01-05 14:59:59"), trade_dates) == pd.Timestamp("2024-01-05")
    assert map_to_trade_date(pd.Timestamp("2024-01-05 15:00:00"), trade_dates) == pd.Timestamp("2024-01-08")
    assert map_to_trade_date(pd.Timestamp("2024-01-05 18:00:00"), trade_dates) == pd.Timestamp("2024-01-08")
    assert map_to_trade_date(pd.Timestamp("2024-01-06 11:00:00"), trade_dates) == pd.Timestamp("2024-01-08")


def test_assign_trade_dates_drops_rows_without_future_calendar() -> None:
    trade_dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-05"]))
    frame = pd.DataFrame(
        [
            {"news_id": "n1", "publish_time": pd.Timestamp("2024-01-05 14:00:00")},
            {"news_id": "n2", "publish_time": pd.Timestamp("2024-01-05 16:00:00")},
        ]
    )

    aligned = assign_trade_dates(frame, trade_dates=trade_dates)

    assert aligned["news_id"].tolist() == ["n1"]
