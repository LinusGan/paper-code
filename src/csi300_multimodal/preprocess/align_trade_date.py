from __future__ import annotations

from datetime import time

import pandas as pd


def build_trade_calendar(market_frame: pd.DataFrame) -> pd.DatetimeIndex:
    calendar = pd.to_datetime(market_frame["trade_date"]).dt.normalize().drop_duplicates().sort_values()
    return pd.DatetimeIndex(calendar)


def _localize_timestamp(timestamp: pd.Timestamp, timezone: str) -> pd.Timestamp:
    if pd.isna(timestamp):
        return timestamp
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        return ts.tz_localize(timezone)
    return ts.tz_convert(timezone)


def map_to_trade_date(
    publish_time: pd.Timestamp,
    trade_dates: pd.DatetimeIndex,
    timezone: str = "Asia/Shanghai",
    close_time: str = "15:00",
) -> pd.Timestamp:
    if pd.isna(publish_time):
        return pd.NaT
    localized = _localize_timestamp(pd.Timestamp(publish_time), timezone)
    current_date = localized.tz_localize(None).normalize()
    cutoff = time.fromisoformat(close_time)
    trade_set = set(trade_dates)

    if current_date in trade_set and localized.timetz().replace(tzinfo=None) < cutoff:
        return current_date

    search_index = trade_dates.searchsorted(current_date, side="right")
    if current_date not in trade_set:
        search_index = trade_dates.searchsorted(current_date, side="left")
    if search_index >= len(trade_dates):
        return pd.NaT
    return pd.Timestamp(trade_dates[search_index]).normalize()


def assign_trade_dates(
    news_frame: pd.DataFrame,
    trade_dates: pd.DatetimeIndex,
    timezone: str = "Asia/Shanghai",
    close_time: str = "15:00",
) -> pd.DataFrame:
    frame = news_frame.copy()
    if frame.empty:
        frame["trade_date"] = pd.Series(dtype="datetime64[ns]")
        frame["publish_date"] = pd.Series(dtype="datetime64[ns]")
        return frame
    frame["trade_date"] = frame["publish_time"].apply(
        map_to_trade_date,
        trade_dates=trade_dates,
        timezone=timezone,
        close_time=close_time,
    )
    frame = frame.dropna(subset=["trade_date"]).sort_values(["trade_date", "publish_time"]).reset_index(drop=True)
    publish_series = pd.to_datetime(frame["publish_time"], utc=False, errors="coerce")
    if getattr(publish_series.dt, "tz", None) is not None:
        publish_series = publish_series.dt.tz_localize(None)
    frame["publish_date"] = publish_series.dt.normalize()
    return frame
