from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from .base import BaseAnomalyDetector


class LOFDetector(BaseAnomalyDetector):

    name = "lof"

    def __init__(
        self,
        n_neighbors: int = 20,
        novelty: bool = True,
        n_jobs: int = 1,
        max_train_samples: int = 8000,
        random_state: int = 0,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.novelty = novelty
        self.n_jobs = n_jobs
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.threshold_ = None
        self.model_ = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray) -> LOFDetector:
        X = np.asarray(X, dtype=float)
        if len(X) > self.max_train_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), size=self.max_train_samples, replace=False)
            X = X[idx]
        n_neighbors = min(self.n_neighbors, max(2, len(X) - 1))
        self.model_ = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # score_samples: higher = more normal → negate
        return -self.model_.score_samples(np.asarray(X, dtype=float))
