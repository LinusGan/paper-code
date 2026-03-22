from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "output_dir": "outputs",
        "market_path": "data/raw/market.parquet",
        "news_path": "data/raw/news.parquet",
        "raw_news_path": "data/raw/news/raw_articles.parquet",
        "cleaned_news_path": "data/processed/news/cleaned_articles.parquet",
        "deduped_news_path": "data/processed/news/deduped_articles.parquet",
        "prepared_market_path": "data/processed/prepared_market.parquet",
        "aligned_news_path": "data/processed/news/aligned_articles.parquet",
        "daily_text_features_path": "data/processed/news/daily_text_features.parquet",
        "encoded_news_path": "data/processed/news/encoded_news.parquet",
        "main_table_path": "data/processed/main_table.parquet",
        "daily_embeddings_path": "data/processed/news/daily_embeddings.parquet",
        "sources_config_path": "configs/sources.yaml",
    },
    "features": {
        "lookback": 20,
        "fillna_value": 0.0,
        "split_dates": {
            "train_end": "2022-12-31",
            "valid_end": "2023-12-31",
            "test_end": "2024-12-31",
        },
        "exclude_regime_inputs": True,
    },
    "text": {
        "backend": "dummy",
        "device": "cpu",
        "batch_size": 16,
        "max_length": 256,
        "embedding_dim": 16,
        "text_column": "content_clean",
        "title_fallback": True,
        "model_name_or_path": "",
        "sentiment_model_name_or_path": "",
        "embedding_model_name_or_path": "",
    },
    "news": {
        "timezone": "Asia/Shanghai",
        "market_close_time": "15:00",
        "request_timeout_seconds": 15,
        "retry_attempts": 3,
        "title_similarity_threshold": 0.95,
        "dedup_window_hours": 24,
    },
    "model": {
        "family": "transformer",
        "modality": "multimodal",
        "task": "cls",
        "fusion": "gate",
        "text_repr": "sentiment",
        "hidden_dim": 64,
        "dropout": 0.1,
        "num_layers": 2,
        "num_heads": 4,
        "xgboost_fallback": True,
    },
    "train": {
        "seed": 42,
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "early_stopping_patience": 5,
        "task_weights": {"cls": 1.0, "reg": 1.0},
        "num_workers": 0,
    },
    "eval": {
        "threshold": 0.5,
        "annualization": 252,
        "transaction_cost": 0.0,
        "do_plot": True,
    },
    "suite": {
        "name": "default_suite",
        "base_config": "configs/default.yaml",
        "axes": {},
    },
}


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_by_dotted_path(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    updated = deepcopy(config)
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected key=value.")
        key, value = raw.split("=", 1)
        set_by_dotted_path(updated, key, yaml.safe_load(value))
    return updated


def load_config(config_path: str | Path | None, overrides: list[str] | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = deep_merge(config, loaded)
    return apply_overrides(config, overrides)


def resolve_path(config: dict[str, Any], key: str) -> Path:
    return Path(config["data"][key]).expanduser()
