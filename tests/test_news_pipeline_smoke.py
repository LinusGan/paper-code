from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from csi300_multimodal.config import load_config
from csi300_multimodal.pipelines.build_dataset import run_build_dataset
from csi300_multimodal.pipelines.crawl_news import run_crawl_news
from csi300_multimodal.pipelines.encode_text import run_encode_text
from csi300_multimodal.pipelines.prepare_market import run_prepare_market
from csi300_multimodal.pipelines.prepare_news import run_prepare_news

from tests.conftest import make_market_frame


def test_news_pipeline_end_to_end_with_example_source(tmp_path: Path) -> None:
    market = make_market_frame(periods=8, start="2024-01-02")
    market_path = tmp_path / "raw" / "market.parquet"
    market_path.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(market_path, index=False)

    fixture_root = Path(__file__).resolve().parents[1] / "data" / "raw" / "news" / "example_source"
    sources_path = tmp_path / "sources.yaml"
    with sources_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "example_finance",
                        "enabled": True,
                        "adapter": "example_source",
                        "start_urls": [str(fixture_root / "listing.html")],
                        "request": {"timeout_seconds": 2},
                        "crawl": {"max_articles": 10},
                        "parser": {
                            "article_link_selector": "a.article-link",
                            "title_selector": "h1.article-title",
                            "publish_time_selector": "time.publish-time",
                            "content_selector": "div.article-content p",
                            "channel_selector": "span.channel",
                        },
                    }
                ]
            },
            handle,
            sort_keys=False,
        )

    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "data": {
                    "market_path": str(market_path),
                    "raw_news_path": str(tmp_path / "raw" / "news" / "raw_articles.parquet"),
                    "cleaned_news_path": str(tmp_path / "processed" / "news" / "cleaned_articles.parquet"),
                    "deduped_news_path": str(tmp_path / "processed" / "news" / "deduped_articles.parquet"),
                    "prepared_market_path": str(tmp_path / "processed" / "prepared_market.parquet"),
                    "aligned_news_path": str(tmp_path / "processed" / "news" / "aligned_articles.parquet"),
                    "daily_text_features_path": str(tmp_path / "processed" / "news" / "daily_text_features.parquet"),
                    "encoded_news_path": str(tmp_path / "processed" / "news" / "encoded_news.parquet"),
                    "main_table_path": str(tmp_path / "processed" / "main_table.parquet"),
                    "daily_embeddings_path": str(tmp_path / "processed" / "news" / "daily_embeddings.parquet"),
                    "sources_config_path": str(sources_path),
                },
                "text": {"backend": "dummy", "embedding_dim": 8, "text_column": "content_clean", "title_fallback": True},
                "news": {
                    "timezone": "Asia/Shanghai",
                    "market_close_time": "15:00",
                    "request_timeout_seconds": 5,
                    "retry_attempts": 2,
                    "title_similarity_threshold": 0.8,
                    "dedup_window_hours": 24,
                },
            },
            handle,
            sort_keys=False,
        )

    config = load_config(config_path)

    run_prepare_market(config)
    run_crawl_news(config)
    run_prepare_news(config)
    run_encode_text(config)
    run_build_dataset(config)

    assert Path(config["data"]["raw_news_path"]).exists()
    assert Path(config["data"]["cleaned_news_path"]).exists()
    assert Path(config["data"]["deduped_news_path"]).exists()
    assert Path(config["data"]["aligned_news_path"]).exists()
    assert Path(config["data"]["daily_text_features_path"]).exists()
    assert Path(config["data"]["encoded_news_path"]).exists()
    assert Path(config["data"]["daily_embeddings_path"]).exists()
    assert Path(config["data"]["main_table_path"]).exists()

    deduped = pd.read_parquet(config["data"]["deduped_news_path"])
    aligned = pd.read_parquet(config["data"]["aligned_news_path"])
    main_table = pd.read_parquet(config["data"]["main_table_path"])

    assert deduped["is_duplicate"].sum() >= 1
    assert aligned["trade_date"].notna().all()
    assert {"news_count", "title_len_mean", "content_len_mean", "sentiment_score"} <= set(main_table.columns)
