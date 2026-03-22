from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from csi300_multimodal.config import load_config
from csi300_multimodal.pipelines.build_dataset import run_build_dataset
from csi300_multimodal.pipelines.crawl_news import run_crawl_news
from csi300_multimodal.pipelines.encode_text import run_encode_text
from csi300_multimodal.pipelines.prepare_market import run_prepare_market
from csi300_multimodal.pipelines.prepare_news import run_prepare_news


def ensure_example_market(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start="2024-01-02", periods=8)
    close = np.linspace(100.0, 104.0, len(dates))
    open_ = close * 0.998
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "open": open_,
            "high": close * 1.01,
            "low": open_ * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 1_100_000, len(dates)),
        }
    )
    frame.to_parquet(path, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(ROOT / "configs" / "default.yaml")
    ensure_example_market(ROOT / config["data"]["market_path"])
    run_prepare_market(config)
    run_crawl_news(config)
    run_prepare_news(config)
    run_encode_text(config)
    run_build_dataset(config)
    print(f"Main table written to {ROOT / config['data']['main_table_path']}")
    print(f"Daily embeddings written to {ROOT / config['data']['daily_embeddings_path']}")


if __name__ == "__main__":
    main()
