from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseAnomalyDetector


class AutoencoderDetector(BaseAnomalyDetector):
    name = "autoencoder"

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (8, 4, 8),
        max_iter: int = 100,
        random_state: int = 0,
        max_train_samples: int = 8000,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.threshold_ = None
        self.scaler_ = StandardScaler()
        self.model_ = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=10,
        )

    def fit(self, X: np.ndarray) -> AutoencoderDetector:
        X = np.asarray(X, dtype=float)
        if len(X) > self.max_train_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), size=self.max_train_samples, replace=False)
            X = X[idx]
        Xs = self.scaler_.fit_transform(X)
        self.model_.fit(Xs, Xs)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        recon = self.model_.predict(Xs)
        return np.mean((Xs - recon) ** 2, axis=1)
