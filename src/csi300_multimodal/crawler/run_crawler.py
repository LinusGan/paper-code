from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from csi300_multimodal.crawler.sources.base import SourceConfig
from csi300_multimodal.crawler.sources.example_source import ExampleSourceAdapter
from csi300_multimodal.data.schemas import NewsArticle
from csi300_multimodal.utils.io import write_table

LOGGER = logging.getLogger(__name__)

ADAPTER_REGISTRY = {
    "example_source": ExampleSourceAdapter,
}


def load_sources_config(path: str | Path) -> list[SourceConfig]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    sources = payload.get("sources", [])
    return [
        SourceConfig(
            name=str(source["name"]),
            adapter=str(source["adapter"]),
            enabled=bool(source.get("enabled", True)),
            base_url=str(source.get("base_url", "")),
            start_urls=list(source.get("start_urls", [])),
            request=dict(source.get("request", {})),
            parser=dict(source.get("parser", {})),
            crawl=dict(source.get("crawl", {})),
            config_dir=config_path.parent,
        )
        for source in sources
    ]


def run_crawler(
    *,
    sources_config_path: str | Path,
    output_path: str | Path,
    news_config: dict[str, Any],
) -> pd.DataFrame:
    all_articles: list[dict[str, Any]] = []
    for source_config in load_sources_config(sources_config_path):
        if not source_config.enabled:
            continue
        adapter_cls = ADAPTER_REGISTRY.get(source_config.adapter)
        if adapter_cls is None:
            LOGGER.warning("Unknown source adapter '%s' for source '%s'. Skipping.", source_config.adapter, source_config.name)
            continue
        LOGGER.info("Crawling source %s with adapter %s", source_config.name, source_config.adapter)
        adapter = adapter_cls(source_config=source_config, news_config=news_config)
        for article in adapter.crawl():
            all_articles.append(article.to_record())

    if all_articles:
        frame = pd.DataFrame.from_records(all_articles, columns=NewsArticle.columns())
        frame = frame.drop_duplicates(subset=["news_id"]).sort_values(["source", "publish_time", "news_id"]).reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=NewsArticle.columns())

    write_table(frame, output_path)
    return frame
