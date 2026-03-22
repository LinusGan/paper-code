from __future__ import annotations

from csi300_multimodal.config import resolve_path
from csi300_multimodal.data.io import load_market_raw
from csi300_multimodal.features.market import compute_market_features
from csi300_multimodal.utils.io import write_table


def run_prepare_market(config: dict) -> None:
    market = load_market_raw(resolve_path(config, "market_path"))
    prepared = compute_market_features(
        market_frame=market,
        split_dates=config["features"]["split_dates"],
        fillna_value=float(config["features"]["fillna_value"]),
    )
    write_table(prepared, resolve_path(config, "prepared_market_path"))
