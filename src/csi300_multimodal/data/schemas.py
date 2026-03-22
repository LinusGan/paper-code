from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from csi300_multimodal.data.contracts import NEWS_ARTICLE_COLUMNS


def build_news_id(source: str, url: str, publish_time: pd.Timestamp, title: str) -> str:
    ts = pd.Timestamp(publish_time) if not pd.isna(publish_time) else pd.Timestamp("1970-01-01")
    payload = "|".join([source.strip(), url.strip(), ts.isoformat(), title.strip()])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"news_{digest[:16]}"


@dataclass(slots=True)
class NewsArticle:
    news_id: str
    source: str
    url: str
    title: str
    publish_time: pd.Timestamp
    content: str
    crawl_time: pd.Timestamp
    channel: str = ""
    raw_title: str = ""
    raw_content: str = ""
    title_clean: str = ""
    content_clean: str = ""
    title_hash: str = ""
    content_hash: str = ""
    is_duplicate: bool = False
    duplicate_reason: str = ""
    canonical_news_id: str = ""
    trade_date: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        self.source = str(self.source).strip()
        self.url = str(self.url).strip()
        self.title = str(self.title).strip()
        self.content = str(self.content or "")
        self.channel = str(self.channel or "").strip()
        self.raw_title = str(self.raw_title or "")
        self.raw_content = str(self.raw_content or "")
        self.title_clean = str(self.title_clean or "")
        self.content_clean = str(self.content_clean or "")
        self.title_hash = str(self.title_hash or "")
        self.content_hash = str(self.content_hash or "")
        self.duplicate_reason = str(self.duplicate_reason or "")
        self.canonical_news_id = str(self.canonical_news_id or self.news_id)
        self.publish_time = pd.Timestamp(self.publish_time)
        self.crawl_time = pd.Timestamp(self.crawl_time)
        self.trade_date = pd.Timestamp(self.trade_date) if self.trade_date is not None and not pd.isna(self.trade_date) else None
        if not self.news_id:
            raise ValueError("news_id is required")
        if not self.source:
            raise ValueError("source is required")
        if not self.url:
            raise ValueError("url is required")
        if not self.title:
            raise ValueError("title is required")
        if pd.isna(self.publish_time):
            raise ValueError("publish_time is required")
        if pd.isna(self.crawl_time):
            raise ValueError("crawl_time is required")

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "NewsArticle":
        payload = {column: record.get(column) for column in NEWS_ARTICLE_COLUMNS}
        source = str(payload.get("source") or "legacy_input").strip()
        title = str(payload.get("title") or "").strip()
        publish_time = pd.Timestamp(payload.get("publish_time"))
        crawl_time = payload.get("crawl_time")
        crawl_ts = pd.Timestamp(crawl_time) if crawl_time is not None and not pd.isna(crawl_time) else publish_time
        news_id = str(payload.get("news_id") or "").strip()
        url = str(payload.get("url") or "").strip()
        if not news_id and title:
            fallback_key = hashlib.sha1(f"{source}|{title}|{publish_time.isoformat()}".encode("utf-8")).hexdigest()[:12]
            placeholder_url = url or f"legacy://{source}/{fallback_key}"
            news_id = build_news_id(source=source, url=placeholder_url, publish_time=publish_time, title=title)
        if not url:
            url = f"legacy://{source}/{news_id}"
        return cls(
            news_id=news_id,
            source=source,
            url=url,
            title=title,
            publish_time=publish_time,
            content=str(payload.get("content") or ""),
            crawl_time=crawl_ts,
            channel=str(payload.get("channel") or ""),
            raw_title=str(payload.get("raw_title") or ""),
            raw_content=str(payload.get("raw_content") or ""),
            title_clean=str(payload.get("title_clean") or ""),
            content_clean=str(payload.get("content_clean") or ""),
            title_hash=str(payload.get("title_hash") or ""),
            content_hash=str(payload.get("content_hash") or ""),
            is_duplicate=bool(payload.get("is_duplicate") or False),
            duplicate_reason=str(payload.get("duplicate_reason") or ""),
            canonical_news_id=str(payload.get("canonical_news_id") or news_id),
            trade_date=payload.get("trade_date"),
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["publish_time"] = pd.Timestamp(self.publish_time)
        record["crawl_time"] = pd.Timestamp(self.crawl_time)
        record["trade_date"] = pd.Timestamp(self.trade_date) if self.trade_date is not None else pd.NaT
        return record

    @staticmethod
    def columns() -> list[str]:
        return list(NEWS_ARTICLE_COLUMNS)
