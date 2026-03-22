from __future__ import annotations

from pathlib import Path

from csi300_multimodal.evaluation.metrics import evaluate_groups, evaluate_predictions
from csi300_multimodal.evaluation.strategy import build_strategy_frame, plot_equity_curve, strategy_metrics
from csi300_multimodal.training.trainer import build_run_dir
from csi300_multimodal.utils.io import dump_json, read_table, write_table


def run_evaluate(config: dict) -> dict:
    run_dir = build_run_dir(config)
    prediction_frame = read_table(run_dir / "predictions.parquet")
    prediction_frame["trade_date"] = prediction_frame["trade_date"].astype("datetime64[ns]")
    threshold = float(config["eval"]["threshold"])

    metrics = evaluate_predictions(prediction_frame, threshold=threshold)
    group_frame = evaluate_groups(
        prediction_frame,
        threshold=threshold,
        group_columns=["trend_regime", "vol_regime", "drawdown_state"],
    )

    strategy_summary: dict[str, dict[str, float]] = {}
    for split, split_frame in prediction_frame.groupby("split"):
        strategy_frame = build_strategy_frame(
            split_frame,
            threshold=threshold,
            transaction_cost=float(config["eval"]["transaction_cost"]),
        )
        strategy_summary[str(split)] = strategy_metrics(
            strategy_frame,
            annualization=int(config["eval"]["annualization"]),
        )
        write_table(strategy_frame, run_dir / f"strategy_{split}.parquet")
        if bool(config["eval"].get("do_plot", True)):
            plot_equity_curve(strategy_frame, run_dir / f"equity_{split}.png")

    payload = {
        "prediction_metrics": metrics,
        "strategy_metrics": strategy_summary,
    }
    write_table(group_frame, run_dir / "group_metrics.csv")
    dump_json(payload, run_dir / "evaluation.json")
    return payload
