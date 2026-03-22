from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from csi300_multimodal.crawler.run_crawler import run_crawler


def test_example_source_crawler_returns_expected_articles(tmp_path: Path) -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "data" / "raw" / "news" / "example_source"
    sources_path = tmp_path / "sources.yaml"
    payload = {
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
    }
    with sources_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    output_path = tmp_path / "raw_articles.parquet"
    frame = run_crawler(sources_config_path=sources_path, output_path=output_path, news_config={"request_timeout_seconds": 5})

    assert output_path.exists()
    assert len(frame) == 3
    assert {"news_id", "source", "url", "title", "publish_time", "content", "crawl_time", "channel"} <= set(frame.columns)
    assert frame["news_id"].nunique() == 3
    reloaded = pd.read_parquet(output_path)
    assert reloaded["source"].eq("example_finance").all()
