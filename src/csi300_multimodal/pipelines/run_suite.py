from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from csi300_multimodal.config import load_config, set_by_dotted_path
from csi300_multimodal.pipelines.evaluate import run_evaluate
from csi300_multimodal.pipelines.train import run_train
from csi300_multimodal.utils.io import ensure_dir, write_table


def _iter_axis_combinations(axes: dict[str, list]) -> list[dict[str, object]]:
    if not axes:
        return [{}]
    keys = list(axes)
    values = [axes[key] for key in keys]
    combinations: list[dict[str, object]] = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def run_suite(config: dict) -> pd.DataFrame:
    base_path = config["suite"].get("base_config")
    base_config = load_config(base_path)
    suite_name = str(config["suite"].get("name", "suite"))
    output_dir = Path(config["data"]["output_dir"]) / "suites" / suite_name
    ensure_dir(output_dir)

    records: list[dict[str, object]] = []
    for index, overrides in enumerate(_iter_axis_combinations(config["suite"].get("axes", {})), start=1):
        experiment = load_config(base_path)
        for key, value in overrides.items():
            set_by_dotted_path(experiment, key, value)
        run_name = f"{suite_name}_{index:02d}"
        experiment["data"]["run_name"] = run_name
        run_train(experiment)
        evaluation = run_evaluate(experiment)
        record = {"run_name": run_name}
        record.update(overrides)
        test_cls = evaluation["prediction_metrics"].get("test", {}).get("classification", {})
        test_reg = evaluation["prediction_metrics"].get("test", {}).get("regression", {})
        test_strategy = evaluation["strategy_metrics"].get("test", {})
        record.update({f"test_cls_{key}": value for key, value in test_cls.items()})
        record.update({f"test_reg_{key}": value for key, value in test_reg.items()})
        record.update({f"test_strategy_{key}": value for key, value in test_strategy.items()})
        records.append(record)

    summary = pd.DataFrame.from_records(records)
    write_table(summary, output_dir / "suite_summary.csv")
    return summary
