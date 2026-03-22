from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from csi300_multimodal.config import load_config
from csi300_multimodal.pipelines.build_dataset import run_build_dataset
from csi300_multimodal.pipelines.encode_text import run_encode_text
from csi300_multimodal.pipelines.evaluate import run_evaluate
from csi300_multimodal.pipelines.prepare_market import run_prepare_market
from csi300_multimodal.pipelines.prepare_news import run_prepare_news
from csi300_multimodal.training.trainer import build_run_dir, train_experiment

from tests.conftest import make_market_frame, make_news_frame, torch_importable, write_test_config


@pytest.mark.parametrize("family", ["xgboost", "lstm", "transformer"])
@pytest.mark.parametrize("task", ["cls", "reg", "multitask"])
def test_end_to_end_training_paths(tmp_path: Path, family: str, task: str) -> None:
    if family in {"lstm", "transformer"} and not torch_importable():
        pytest.skip("torch is not importable in this environment")
    market = make_market_frame(periods=90)
    news = make_news_frame(market)
    market_path = tmp_path / "raw" / "market.parquet"
    news_path = tmp_path / "raw" / "news.parquet"
    market_path.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(market_path, index=False)
    news.to_parquet(news_path, index=False)

    config_path = write_test_config(tmp_path, market_path, news_path)
    config = load_config(config_path)
    config["model"]["family"] = family
    config["model"]["task"] = task
    config["data"]["run_name"] = f"{family}_{task}"

    run_prepare_market(config)
    run_prepare_news(config)
    run_encode_text(config)
    run_build_dataset(config)
    result = train_experiment(config)
    evaluation = run_evaluate(config)

    run_dir = build_run_dir(config)
    assert result.predictions_path.exists()
    assert (run_dir / "evaluation.json").exists()
    assert (run_dir / "group_metrics.csv").exists()
    pred = pd.read_parquet(result.predictions_path)
    assert {"trade_date", "split", "next_ret_1", "next_oc_return"} <= set(pred.columns)
    assert "test" in evaluation["strategy_metrics"]
