from __future__ import annotations

from csi300_multimodal.models.xgb import TabularForecastModel


def build_model(config: dict, input_dims: dict[str, int]):
    family = config["model"]["family"]
    if family == "xgboost":
        return TabularForecastModel(
            task=config["model"]["task"],
            use_fallback=bool(config["model"].get("xgboost_fallback", False)),
        )
    if family in {"lstm", "transformer"}:
        from csi300_multimodal.models.deep import DeepForecastModel

        return DeepForecastModel(config=config, input_dims=input_dims)
    raise ValueError(f"Unsupported model family: {family}")
