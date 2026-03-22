from __future__ import annotations

import hashlib
import html
import re

import pandas as pd
from bs4 import BeautifulSoup


WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]+")


def _strip_markup(text: str) -> str:
    value = html.unescape(str(text or ""))
    if "<" in value and ">" in value:
        value = BeautifulSoup(value, "lxml").get_text(" ", strip=True)
    value = value.replace("\xa0", " ").replace("\u3000", " ")
    value = CONTROL_RE.sub("", value)
    return WHITESPACE_RE.sub(" ", value).strip()


def _hash_text(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()


def clean_news_frame(news_frame: pd.DataFrame, title_fallback: bool = True) -> pd.DataFrame:
    frame = news_frame.copy()
    if frame.empty:
        for column in ["raw_title", "raw_content", "title_clean", "content_clean", "title_hash", "content_hash"]:
            frame[column] = pd.Series(dtype=object)
        return frame

    frame["raw_title"] = frame.get("raw_title", frame["title"]).fillna("")
    frame["raw_content"] = frame.get("raw_content", frame["content"]).fillna("")
    frame["title_clean"] = frame["title"].fillna("").map(_strip_markup)
    frame["content_clean"] = frame["content"].fillna("").map(_strip_markup)
    if title_fallback:
        missing_content = frame["content_clean"].eq("")
        frame.loc[missing_content, "content_clean"] = frame.loc[missing_content, "title_clean"]
    frame["title_hash"] = frame["title_clean"].map(_hash_text)
    frame["content_hash"] = frame["content_clean"].map(_hash_text)
    return frame
