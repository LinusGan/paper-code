from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_strategy_frame(frame: pd.DataFrame, threshold: float, transaction_cost: float) -> pd.DataFrame:
    strategy = frame.copy().sort_values("trade_date").reset_index(drop=True)
    if strategy["pred_cls_score"].notna().any():
        strategy["signal"] = (strategy["pred_cls_score"] >= threshold).astype(int)
    elif strategy["pred_reg"].notna().any():
        strategy["signal"] = (strategy["pred_reg"] > 0.0).astype(int)
    else:
        strategy["signal"] = 0
    strategy["prev_signal"] = strategy["signal"].shift(1).fillna(0).astype(int)
    strategy["turnover"] = (strategy["signal"] - strategy["prev_signal"]).abs()
    strategy["strategy_return"] = strategy["signal"] * strategy["next_oc_return"] - strategy["turnover"] * transaction_cost
    strategy["equity_curve"] = (1.0 + strategy["strategy_return"]).cumprod()
    return strategy


def strategy_metrics(frame: pd.DataFrame, annualization: int) -> dict[str, float]:
    returns = frame["strategy_return"].to_numpy(dtype=float)
    equity = frame["equity_curve"].to_numpy(dtype=float)
    if len(returns) == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
    cumulative_return = float(equity[-1] - 1.0)
    annualized_return = float((equity[-1] ** (annualization / len(equity))) - 1.0) if equity[-1] > 0 else -1.0
    volatility = returns.std(ddof=0)
    sharpe = float((returns.mean() / volatility) * np.sqrt(annualization)) if volatility > 0 else 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0
    win_rate = float((returns > 0).mean())
    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": win_rate,
    }


def plot_equity_curve(frame: pd.DataFrame, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(frame["trade_date"], frame["equity_curve"], label="strategy")
    plt.title("Equity Curve")
    plt.xlabel("trade_date")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(target)
    plt.close()
    return target
