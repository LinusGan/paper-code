from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def torch_importable() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


def make_market_frame(periods: int = 90, start: str = "2022-10-03") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=periods)
    close = np.linspace(100.0, 130.0, periods) + np.sin(np.arange(periods)) * 2.0
    open_ = close * (1.0 + 0.001 * np.cos(np.arange(periods)))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = 1_000_000 + np.arange(periods) * 1_000
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def make_news_frame(market: pd.DataFrame) -> pd.DataFrame:
    records = []
    for index, row in market.iloc[::2].reset_index(drop=True).iterrows():
        publish_day = pd.Timestamp(row["trade_date"])
        records.append(
            {
                "news_id": f"n{index:03d}",
                "source": "unit_test",
                "url": f"https://example.test/news/{index:03d}",
                "publish_time": publish_day + pd.Timedelta(hours=10),
                "title": "market rise and profit",
                "content": "profit growth and optimistic outlook",
                "crawl_time": publish_day + pd.Timedelta(hours=10, minutes=5),
                "channel": "macro",
            }
        )
        records.append(
            {
                "news_id": f"na{index:03d}",
                "source": "unit_test",
                "url": f"https://example.test/news/after/{index:03d}",
                "publish_time": publish_day + pd.Timedelta(hours=16),
                "title": "risk warning",
                "content": "risk warning and possible decline",
                "crawl_time": publish_day + pd.Timedelta(hours=16, minutes=5),
                "channel": "macro",
            }
        )
    return pd.DataFrame.from_records(records)


def write_test_config(tmp_path: Path, market_path: Path, news_path: Path) -> Path:
    config = {
        "data": {
            "market_path": str(market_path),
            "news_path": str(news_path),
            "raw_news_path": str(tmp_path / "raw" / "news" / "raw_articles.parquet"),
            "cleaned_news_path": str(tmp_path / "processed" / "news" / "cleaned_articles.parquet"),
            "deduped_news_path": str(tmp_path / "processed" / "news" / "deduped_articles.parquet"),
            "prepared_market_path": str(tmp_path / "processed" / "prepared_market.parquet"),
            "aligned_news_path": str(tmp_path / "processed" / "news" / "aligned_articles.parquet"),
            "daily_text_features_path": str(tmp_path / "processed" / "news" / "daily_text_features.parquet"),
            "encoded_news_path": str(tmp_path / "processed" / "news" / "encoded_news.parquet"),
            "main_table_path": str(tmp_path / "processed" / "main_table.parquet"),
            "daily_embeddings_path": str(tmp_path / "processed" / "news" / "daily_embeddings.parquet"),
            "output_dir": str(tmp_path / "outputs"),
            "sources_config_path": str(tmp_path / "sources.yaml"),
        },
        "features": {
            "lookback": 5,
            "fillna_value": 0.0,
            "split_dates": {
                "train_end": "2022-11-30",
                "valid_end": "2022-12-30",
                "test_end": "2023-02-28",
            },
            "exclude_regime_inputs": True,
        },
        "text": {
            "backend": "dummy",
            "device": "cpu",
            "batch_size": 8,
            "max_length": 64,
            "embedding_dim": 8,
            "text_column": "content_clean",
            "title_fallback": True,
        },
        "news": {
            "timezone": "Asia/Shanghai",
            "market_close_time": "15:00",
            "request_timeout_seconds": 5,
            "retry_attempts": 2,
            "title_similarity_threshold": 0.95,
            "dedup_window_hours": 24,
        },
        "model": {
            "family": "transformer",
            "modality": "multimodal",
            "task": "cls",
            "fusion": "gate",
            "text_repr": "sentiment",
            "hidden_dim": 16,
            "dropout": 0.1,
            "num_layers": 1,
            "num_heads": 2,
            "xgboost_fallback": True,
        },
        "train": {
            "seed": 7,
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "early_stopping_patience": 2,
            "task_weights": {"cls": 1.0, "reg": 1.0},
            "num_workers": 0,
        },
        "eval": {
            "threshold": 0.5,
            "annualization": 252,
            "transaction_cost": 0.0,
            "do_plot": False,
        },
    }
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path
