from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def _build_classifier(allow_fallback: bool):
    if HAS_XGBOOST:
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
        )
    if not allow_fallback:
        raise ImportError("xgboost is not installed and fallback is disabled.")
    return GradientBoostingClassifier(random_state=42)


def _build_regressor(allow_fallback: bool):
    if HAS_XGBOOST:
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        )
    if not allow_fallback:
        raise ImportError("xgboost is not installed and fallback is disabled.")
    return GradientBoostingRegressor(random_state=42)


@dataclass
class TabularForecastModel:
    task: str
    use_fallback: bool = False

    def __post_init__(self) -> None:
        self.cls_model = None
        self.reg_model = None
        if self.task in {"cls", "multitask"}:
            self.cls_model = _build_classifier(self.use_fallback)
        if self.task in {"reg", "multitask"}:
            self.reg_model = _build_regressor(self.use_fallback)

    def fit(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray) -> None:
        if self.cls_model is not None:
            unique = np.unique(y_cls[~np.isnan(y_cls)])
            if len(unique) < 2:
                self.cls_model = DummyClassifier(strategy="constant", constant=int(unique[0] if len(unique) else 0))
            self.cls_model.fit(X, y_cls.astype(int))
        if self.reg_model is not None:
            if np.nanstd(y_reg) == 0.0:
                self.reg_model = DummyRegressor(strategy="constant", constant=float(np.nanmean(y_reg)))
            self.reg_model.fit(X, y_reg.astype(float))

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        outputs: dict[str, np.ndarray] = {}
        if self.cls_model is not None:
            if hasattr(self.cls_model, "predict_proba"):
                outputs["pred_cls_score"] = self.cls_model.predict_proba(X)[:, 1]
            else:
                outputs["pred_cls_score"] = self.cls_model.predict(X).astype(float)
            outputs["pred_cls_label"] = (outputs["pred_cls_score"] >= 0.5).astype(int)
        if self.reg_model is not None:
            outputs["pred_reg"] = self.reg_model.predict(X)
        return outputs
