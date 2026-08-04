
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .base import BaseAnomalyDetector


class OneClassSVMDetector(BaseAnomalyDetector):

    name = "one_class_svm"

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str | float = "scale",
        random_state: int = 0,
        max_train_samples: int = 5000,
    ) -> None:
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.threshold_ = None
        self.scaler_ = StandardScaler()
        self.model_ = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X: np.ndarray) -> OneClassSVMDetector:
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
        # decision_function: higher = more normal → negate
        return -self.model_.decision_function(Xs)
