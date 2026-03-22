from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FrameStandardizer:
    columns: list[str]
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> None:
        for column in self.columns:
            mean = float(frame[column].mean())
            std = float(frame[column].std(ddof=0))
            self.means[column] = mean
            self.stds[column] = std if std > 0 else 1.0

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        updated = frame.copy()
        for column in self.columns:
            updated[column] = (updated[column] - self.means[column]) / self.stds[column]
        return updated

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"means": self.means, "stds": self.stds}
