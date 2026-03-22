from __future__ import annotations

from difflib import SequenceMatcher
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


def _normalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    parts = urlsplit(value)
    if parts.scheme == "legacy":
        return value
    normalized_path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), normalized_path, "", ""))


def _content_length(frame: pd.DataFrame) -> pd.Series:
    return frame["content_clean"].fillna("").str.len()


def _select_canonical(group: pd.DataFrame) -> str:
    ordered = group.assign(_content_len=_content_length(group)).sort_values(
        ["publish_time", "_content_len", "news_id"],
        ascending=[True, False, True],
    )
    return str(ordered.iloc[0]["news_id"])


def _assign_duplicates(frame: pd.DataFrame, mask: pd.Series, canonical_id: str, reason: str) -> None:
    duplicate_mask = mask & frame["news_id"].ne(canonical_id) & (~frame["is_duplicate"])
    frame.loc[duplicate_mask, "is_duplicate"] = True
    frame.loc[duplicate_mask, "duplicate_reason"] = reason
    frame.loc[duplicate_mask, "canonical_news_id"] = canonical_id
    frame.loc[frame["news_id"].eq(canonical_id), "canonical_news_id"] = canonical_id


def deduplicate_news_frame(
    news_frame: pd.DataFrame,
    title_similarity_threshold: float = 0.95,
    dedup_window_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = news_frame.copy()
    if frame.empty:
        frame["url_normalized"] = pd.Series(dtype=object)
        frame["is_duplicate"] = pd.Series(dtype=bool)
        frame["duplicate_reason"] = pd.Series(dtype=object)
        frame["canonical_news_id"] = pd.Series(dtype=object)
        return frame, frame

    frame["publish_time"] = pd.to_datetime(frame["publish_time"], utc=False, errors="coerce")
    frame = frame.sort_values(["publish_time", "news_id"]).reset_index(drop=True)
    frame["url_normalized"] = frame["url"].fillna("").map(_normalize_url)
    frame["exact_key"] = frame["title_hash"].fillna("") + "|" + frame["content_hash"].fillna("")
    frame["is_duplicate"] = False
    frame["duplicate_reason"] = ""
    frame["canonical_news_id"] = frame["news_id"]

    non_legacy_url = frame["url_normalized"].str.startswith("legacy://") == False
    for _, group in frame.loc[non_legacy_url & frame["url_normalized"].ne("")].groupby("url_normalized"):
        if len(group) < 2:
            continue
        canonical_id = _select_canonical(group)
        _assign_duplicates(frame, frame.index.isin(group.index), canonical_id, "exact_url")

    for _, group in frame.loc[frame["exact_key"].ne("|")].groupby("exact_key"):
        if len(group) < 2:
            continue
        canonical_id = _select_canonical(group)
        _assign_duplicates(frame, frame.index.isin(group.index), canonical_id, "exact_text")

    canonical_rows = frame.loc[~frame["is_duplicate"]].sort_values(["publish_time", "news_id"]).copy()
    window = pd.Timedelta(hours=dedup_window_hours)
    for row in canonical_rows.itertuples(index=False):
        if bool(frame.loc[frame["news_id"].eq(row.news_id), "is_duplicate"].iloc[0]):
            continue
        current_time = pd.Timestamp(row.publish_time)
        candidates = frame.loc[
            (~frame["is_duplicate"])
            & frame["news_id"].ne(row.news_id)
            & frame["publish_time"].between(current_time, current_time + window, inclusive="both")
        ]
        for candidate in candidates.itertuples(index=False):
            similarity = SequenceMatcher(None, str(row.title_clean), str(candidate.title_clean)).ratio()
            if similarity < title_similarity_threshold:
                continue
            canonical_id = _select_canonical(frame.loc[frame["news_id"].isin([row.news_id, candidate.news_id])])
            duplicate_id = candidate.news_id if canonical_id == row.news_id else row.news_id
            frame.loc[frame["news_id"].eq(duplicate_id), "is_duplicate"] = True
            frame.loc[frame["news_id"].eq(duplicate_id), "duplicate_reason"] = "near_title"
            frame.loc[frame["news_id"].eq(duplicate_id), "canonical_news_id"] = canonical_id
            frame.loc[frame["news_id"].eq(canonical_id), "canonical_news_id"] = canonical_id

    deduped = frame.sort_values(["publish_time", "news_id"]).reset_index(drop=True)
    unique = deduped.loc[~deduped["is_duplicate"]].reset_index(drop=True)
    return deduped, unique
