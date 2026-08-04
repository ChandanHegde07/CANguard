

from __future__ import annotations

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

from .base import BaseAnomalyDetector


class EllipticEnvelopeDetector(BaseAnomalyDetector):

    name = "elliptic_envelope"

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 0,
        support_fraction: float | None = None,
        max_train_samples: int = 8000,
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.support_fraction = support_fraction
        self.max_train_samples = max_train_samples
        self.threshold_ = None
        self.scaler_ = StandardScaler()
        self.model_ = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state,
            support_fraction=support_fraction,
        )

    def fit(self, X: np.ndarray) -> EllipticEnvelopeDetector:
        X = np.asarray(X, dtype=float)
        if len(X) > self.max_train_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), size=self.max_train_samples, replace=False)
            X = X[idx]
        Xs = self.scaler_.fit_transform(X)
        self.model_.fit(Xs)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        # score_samples: higher = more normal → negate
        return -self.model_.score_samples(Xs)
