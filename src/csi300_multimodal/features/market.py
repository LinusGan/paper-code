from __future__ import annotations

import numpy as np
import pandas as pd

from csi300_multimodal.data.contracts import MARKET_FEATURE_COLUMNS


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def assign_split(trade_date: pd.Series, split_dates: dict[str, str]) -> pd.Series:
    train_end = pd.Timestamp(split_dates["train_end"])
    valid_end = pd.Timestamp(split_dates["valid_end"])
    labels = np.where(
        trade_date <= train_end,
        "train",
        np.where(trade_date <= valid_end, "valid", "test"),
    )
    labels = pd.Series(labels, index=trade_date.index)
    labels[trade_date > pd.Timestamp(split_dates["test_end"])] = "holdout"
    return labels


def compute_market_features(
    market_frame: pd.DataFrame,
    split_dates: dict[str, str],
    fillna_value: float = 0.0,
) -> pd.DataFrame:
    frame = market_frame.copy().sort_values("trade_date").reset_index(drop=True)
    close = frame["close"]
    volume = frame["volume"]

    for window in [1, 2, 3, 5, 10, 20]:
        frame[f"ret_{window}"] = close.pct_change(window)

    frame["volatility_5"] = frame["ret_1"].rolling(5, min_periods=5).std()
    frame["volatility_10"] = frame["ret_1"].rolling(10, min_periods=10).std()
    frame["volatility_20"] = frame["ret_1"].rolling(20, min_periods=20).std()
    frame["hl_range"] = (frame["high"] - frame["low"]) / close.replace(0.0, np.nan)
    frame["oc_return"] = (frame["close"] - frame["open"]) / frame["open"].replace(0.0, np.nan)
    frame["gap_return"] = frame["open"] / frame["close"].shift(1).replace(0.0, np.nan) - 1.0

    for window in [5, 10, 20]:
        frame[f"ma_{window}"] = close.rolling(window, min_periods=window).mean()
        frame[f"ma_gap_{window}"] = close / frame[f"ma_{window}"].replace(0.0, np.nan) - 1.0

    frame["rsi_14"] = compute_rsi(close, 14)
    frame["momentum_5"] = close.diff(5)
    frame["momentum_10"] = close.diff(10)

    frame["volume_change_1"] = volume.pct_change(1)
    frame["volume_change_5"] = volume.pct_change(5)
    frame["price_volume_corr_5"] = frame["ret_1"].rolling(5, min_periods=5).corr(frame["volume_change_1"])
    frame["price_volume_corr_10"] = frame["ret_1"].rolling(10, min_periods=10).corr(frame["volume_change_1"])

    trend_signal = frame["ma_gap_20"].fillna(0.0)
    frame["trend_regime"] = (trend_signal > 0.0).astype(int)

    vol_threshold = frame["volatility_20"].rolling(60, min_periods=20).median()
    frame["vol_regime"] = (frame["volatility_20"] > vol_threshold).fillna(False).astype(int)

    rolling_peak = close.cummax()
    drawdown = close / rolling_peak.replace(0.0, np.nan) - 1.0
    frame["drawdown_state"] = (drawdown <= -0.1).fillna(False).astype(int)

    frame["next_ret_1"] = close.shift(-1) / close.replace(0.0, np.nan) - 1.0
    frame["next_oc_return"] = frame["close"].shift(-1) / frame["open"].shift(-1).replace(0.0, np.nan) - 1.0
    frame["y_cls"] = (frame["next_ret_1"] > 0.0).astype(float)
    frame.loc[frame["next_ret_1"].isna(), "y_cls"] = np.nan
    frame["split"] = assign_split(frame["trade_date"], split_dates)

    frame[MARKET_FEATURE_COLUMNS] = frame[MARKET_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(fillna_value)
    return frame
