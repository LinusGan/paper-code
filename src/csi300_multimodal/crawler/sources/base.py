from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from csi300_multimodal.crawler.utils.parser_utils import extract_links, parse_html, select_attr, select_text
from csi300_multimodal.crawler.utils.request_utils import fetch_text, resolve_resource
from csi300_multimodal.data.schemas import NewsArticle, build_news_id

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SourceConfig:
    name: str
    adapter: str
    enabled: bool = True
    base_url: str = ""
    start_urls: list[str] = field(default_factory=list)
    request: dict[str, Any] = field(default_factory=dict)
    parser: dict[str, Any] = field(default_factory=dict)
    crawl: dict[str, Any] = field(default_factory=dict)
    config_dir: Path = field(default_factory=lambda: Path.cwd())


class BaseSourceAdapter:
    def __init__(self, source_config: SourceConfig, news_config: dict[str, Any]) -> None:
        self.source_config = source_config
        self.news_config = news_config

    def crawl(self) -> list[NewsArticle]:
        raise NotImplementedError

    @property
    def timeout_seconds(self) -> int:
        return int(self.source_config.request.get("timeout_seconds") or self.news_config.get("request_timeout_seconds", 15))

    @property
    def base_dir(self) -> Path:
        return self.source_config.config_dir

    def fetch(self, resource: str) -> str:
        return fetch_text(
            resource,
            timeout_seconds=self.timeout_seconds,
            retry_attempts=int(self.news_config.get("retry_attempts", 3)),
            base_dir=self.base_dir,
            headers={"User-Agent": "csi300-multimodal-news-pipeline/0.1"},
        )

    def resolve_resource(self, resource: str) -> str:
        return resolve_resource(resource, base_dir=self.base_dir)


class HtmlListDetailSourceAdapter(BaseSourceAdapter):
    def crawl(self) -> list[NewsArticle]:
        parser = self.source_config.parser
        article_links: list[str] = []
        base_url = self.resolve_resource(
            self.source_config.base_url or (self.source_config.start_urls[0] if self.source_config.start_urls else "")
        )
        for start_url in self.source_config.start_urls:
            resolved_start_url = self.resolve_resource(start_url)
            listing_html = self.fetch(resolved_start_url)
            listing_soup = parse_html(listing_html)
            article_links.extend(
                extract_links(
                    listing_soup,
                    parser["article_link_selector"],
                    base_url=base_url or resolved_start_url,
                )
            )

        max_articles = int(self.source_config.crawl.get("max_articles", len(article_links) or 0))
        unique_links = []
        seen: set[str] = set()
        for link in article_links:
            if link in seen:
                continue
            seen.add(link)
            unique_links.append(link)
        if max_articles > 0:
            unique_links = unique_links[:max_articles]

        articles: list[NewsArticle] = []
        for link in unique_links:
            try:
                article_html = self.fetch(link)
                article = self.parse_article(link, article_html)
            except Exception as exc:
                LOGGER.warning("Failed to parse article %s from source %s: %s", link, self.source_config.name, exc)
                continue
            articles.append(article)
        return articles

    def parse_article(self, article_url: str, html: str) -> NewsArticle:
        parser = self.source_config.parser
        soup = parse_html(html)
        title = select_text(soup, parser["title_selector"])
        publish_raw = select_attr(soup, parser["publish_time_selector"], "datetime") or select_text(
            soup, parser["publish_time_selector"]
        )
        content = select_text(soup, parser["content_selector"], separator="\n")
        channel_selector = parser.get("channel_selector", "")
        channel = select_text(soup, channel_selector) if channel_selector else ""
        publish_time = pd.to_datetime(publish_raw, utc=False)
        crawl_time = pd.Timestamp.utcnow().tz_localize(None)
        news_id = build_news_id(self.source_config.name, article_url, publish_time, title)
        return NewsArticle(
            news_id=news_id,
            source=self.source_config.name,
            url=article_url,
            title=title,
            publish_time=publish_time,
            content=content,
            crawl_time=crawl_time,
            channel=channel,
        )
