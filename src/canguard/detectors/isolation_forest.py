
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF

from .base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 0,
        n_jobs: int = 1,
        contamination="auto",
        max_samples="auto",
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.contamination = contamination
        self.max_samples = max_samples
        self.threshold_ = None
        self.model_ = SklearnIF(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            contamination=contamination,
            max_samples=max_samples,
        )

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        self.model_.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.score_samples(X)
