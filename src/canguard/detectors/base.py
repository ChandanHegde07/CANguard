
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BaseAnomalyDetector(ABC):

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray) -> BaseAnomalyDetector:
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        thr = self.threshold_ if threshold is None else threshold
        if thr is None:
            raise RuntimeError("No threshold set; pass threshold= or calibrate first.")
        return (self.score_samples(X) >= float(thr)).astype(int)

    def predict_with_threshold(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return self.predict(X, threshold=threshold)

    def set_threshold(self, threshold: float) -> None:
        self.threshold_ = float(threshold)

    def save(self, path: str | Path) -> Path:
        """Serialize detector to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> BaseAnomalyDetector:
        obj = joblib.load(path)
        if not isinstance(obj, BaseAnomalyDetector):
            raise TypeError(f"Expected BaseAnomalyDetector, got {type(obj)}")
        return obj

    @property
    def feature_importances_(self) -> Any:
        return None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Default attribute so predict() can check for calibration.
        if not hasattr(cls, "threshold_"):
            cls.threshold_ = None  # type: ignore[attr-defined]
