"""Isolation Forest anomaly detector.

Wraps ``sklearn.ensemble.IsolationForest`` with the CANguard convention:
anomaly score is the negative ``score_samples`` output, so higher = more
anomalous. Training/sampling protocol mirrors ``pird_hcrl.ipynb`` exactly
(n_estimators=200, random_state=0, contamination='auto').
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF

from .base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest operating on feature vectors (residual or raw)."""

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 0,
        n_jobs: int = -1,
        contamination="auto",
    ) -> None:
        """Initialize the underlying sklearn IsolationForest.

        Parameters
        ----------
        n_estimators : int
            Number of isolation trees (default 200).
        random_state : int
            Random seed for reproducibility (default 0).
        n_jobs : int
            Parallel jobs; -1 uses all cores.
        contamination : float | 'auto'
            Expected proportion of outliers; 'auto' matches notebook protocol.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.contamination = contamination
        self.model_ = SklearnIF(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            contamination=contamination,
        )

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        self.model_.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # Negate so higher = more anomalous (notebook convention).
        return -self.model_.score_samples(X)

    @property
    def feature_importances_(self):
        # sklearn IsolationForest does not expose feature importances.
        return None
