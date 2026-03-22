from __future__ import annotations

import pandas as pd

from csi300_multimodal.preprocess.build_daily_text_features import build_daily_text_features
from csi300_multimodal.preprocess.clean_news import clean_news_frame
from csi300_multimodal.preprocess.deduplicate import deduplicate_news_frame


def test_clean_news_preserves_raw_fields_and_applies_title_fallback() -> None:
    frame = pd.DataFrame(
        [
            {
                "news_id": "n1",
                "source": "test",
                "url": "https://example.test/1",
                "title": "<h1> 银行 板块 走强 </h1>",
                "publish_time": pd.Timestamp("2024-01-03 10:00:00"),
                "content": "",
                "crawl_time": pd.Timestamp("2024-01-03 10:01:00"),
                "channel": "macro",
            }
        ]
    )

    cleaned = clean_news_frame(frame, title_fallback=True)

    assert cleaned.loc[0, "raw_title"] == "<h1> 银行 板块 走强 </h1>"
    assert cleaned.loc[0, "title_clean"] == "银行 板块 走强"
    assert cleaned.loc[0, "content_clean"] == "银行 板块 走强"
    assert cleaned.loc[0, "title_hash"]


def test_deduplicate_news_flags_exact_and_near_duplicates_and_daily_stats() -> None:
    frame = pd.DataFrame(
        [
            {
                "news_id": "n1",
                "source": "test",
                "url": "https://example.test/1",
                "title": "政策支持提振银行板块",
                "publish_time": pd.Timestamp("2024-01-03 14:30:00"),
                "content": "监管信号改善银行预期",
                "crawl_time": pd.Timestamp("2024-01-03 14:31:00"),
                "channel": "macro",
            },
            {
                "news_id": "n2",
                "source": "test",
                "url": "https://example.test/2",
                "title": "政策支持提振银行板块快讯",
                "publish_time": pd.Timestamp("2024-01-03 14:45:00"),
                "content": "监管信号改善银行预期",
                "crawl_time": pd.Timestamp("2024-01-03 14:46:00"),
                "channel": "macro",
            },
            {
                "news_id": "n3",
                "source": "test",
                "url": "https://example.test/1",
                "title": "政策支持提振银行板块",
                "publish_time": pd.Timestamp("2024-01-03 14:50:00"),
                "content": "监管信号改善银行预期",
                "crawl_time": pd.Timestamp("2024-01-03 14:51:00"),
                "channel": "macro",
            },
        ]
    )

    cleaned = clean_news_frame(frame)
    deduped, unique = deduplicate_news_frame(cleaned, title_similarity_threshold=0.8, dedup_window_hours=24)

    assert len(unique) == 1
    assert deduped["is_duplicate"].sum() == 2
    assert set(deduped.loc[deduped["is_duplicate"], "duplicate_reason"]) == {"exact_url", "near_title"}

    unique = unique.assign(trade_date=pd.Timestamp("2024-01-03"))
    daily = build_daily_text_features(unique, trade_dates=[pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")])
    assert daily.loc[daily["trade_date"].eq(pd.Timestamp("2024-01-03")), "news_count"].iloc[0] == 1
    assert daily.loc[daily["trade_date"].eq(pd.Timestamp("2024-01-04")), "finbert_neu_mean"].iloc[0] == 1.0
