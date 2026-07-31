"""Abstract anomaly-detector contract.

Defines the interface shared by all detectors so the evaluation pipeline and
experiment runners are detector-agnostic. An "anomaly score" is defined to be
*higher* for more anomalous samples (consistent with the residual pipeline).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAnomalyDetector(ABC):
    """Anomaly detector operating on residual (or raw) feature vectors.

    Subclasses implement :meth:`fit` and :meth:`score_samples`. Callers convert
    scores to hard predictions via a threshold using
    :meth:`predict_with_threshold`.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> BaseAnomalyDetector:
        """Fit the detector on ``X``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature vectors. Detectors typically expect anomaly-free
            (normal-only) input.
        """
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return an anomaly score for each sample (higher = more anomalous).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        """
        raise NotImplementedError

    def predict_with_threshold(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return 0/1 predictions given a decision threshold on the score."""
        return (self.score_samples(X) >= threshold).astype(int)

    @property
    @abstractmethod
    def feature_importances_(self):
        """Feature importances, or None if the model does not expose any."""
        raise NotImplementedError
