"""Abstract anomaly-detector contract.

Defines the interface shared by all detectors so the evaluation pipeline and
experiment runners are detector-agnostic. An "anomaly score" is defined to be
*higher* for more anomalous samples (consistent with the residual pipeline).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BaseAnomalyDetector(ABC):
    """Anomaly detector operating on residual (or raw) feature vectors.

    Subclasses implement :meth:`fit` and :meth:`score_samples`. Callers convert
    scores to hard predictions via a threshold using :meth:`predict`.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray) -> BaseAnomalyDetector:
        """Fit the detector on ``X`` (typically normal-only)."""
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        """Alias for :meth:`score_samples`."""
        return self.score_samples(X)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Return 0/1 predictions.

        If ``threshold`` is None, uses ``self.threshold_`` set after calibration.
        """
        thr = self.threshold_ if threshold is None else threshold
        if thr is None:
            raise RuntimeError("No threshold set; pass threshold= or calibrate first.")
        return (self.score_samples(X) >= float(thr)).astype(int)

    def predict_with_threshold(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Backward-compatible alias for :meth:`predict`."""
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
        """Load a detector previously written by :meth:`save`."""
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
