from __future__ import annotations

import pandas as pd

from csi300_multimodal.data.calendar import map_to_trade_date


def test_calendar_alignment_handles_before_after_close_and_weekend() -> None:
    trade_dates = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    )
    before_close = pd.Timestamp("2024-01-05 10:00:00")
    after_close = pd.Timestamp("2024-01-05 16:00:00")
    weekend = pd.Timestamp("2024-01-06 11:00:00")

    assert map_to_trade_date(before_close, trade_dates) == pd.Timestamp("2024-01-05")
    assert map_to_trade_date(after_close, trade_dates) == pd.Timestamp("2024-01-08")
    assert map_to_trade_date(weekend, trade_dates) == pd.Timestamp("2024-01-08")
