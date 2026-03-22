from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(frame: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=index)
        return path
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=index)
        return path
    raise ValueError(f"Unsupported table format: {path}")


def dump_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def dump_yaml(payload: dict[str, Any], path: str | Path) -> Path:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path
