from __future__ import annotations

import logging

from csi300_multimodal.config import resolve_path
from csi300_multimodal.crawler.run_crawler import run_crawler

LOGGER = logging.getLogger(__name__)


def run_crawl_news(config: dict) -> None:
    LOGGER.info("Running news crawl")
    run_crawler(
        sources_config_path=resolve_path(config, "sources_config_path"),
        output_path=resolve_path(config, "raw_news_path"),
        news_config=config.get("news", {}),
    )
