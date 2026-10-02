
from __future__ import annotations

import numpy as np

from .base import BaseAnomalyDetector


class HBOSDetector(BaseAnomalyDetector):

    name = "hbos"

    def __init__(self, n_bins: int = 10, tol: float = 0.5, alpha: float = 0.1) -> None:
        self.n_bins = n_bins
        self.tol = tol
        self.alpha = alpha
        self.threshold_ = None
        self.histograms_: list[tuple[np.ndarray, np.ndarray]] | None = None

    def fit(self, X: np.ndarray) -> HBOSDetector:
        X = np.asarray(X, dtype=float)
        self.histograms_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            # Handle constant columns.
            if np.nanstd(col) < 1e-12:
                edges = np.array([col.min() - 1.0, col.max() + 1.0])
                dens = np.array([1.0])
            else:
                counts, edges = np.histogram(col, bins=self.n_bins)
                dens = counts.astype(float) + self.alpha
                dens = dens / dens.max()
            self.histograms_.append((edges, dens))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.histograms_ is None:
            raise RuntimeError("HBOSDetector must be fit before scoring.")
        X = np.asarray(X, dtype=float)
        scores = np.zeros(len(X), dtype=float)
        for j, (edges, dens) in enumerate(self.histograms_):
            # Bin index for each sample.
            idx = np.searchsorted(edges, X[:, j], side="right") - 1
            idx = np.clip(idx, 0, len(dens) - 1)
            # Low density → high anomaly
            dens_j = dens[idx]
            scores += -np.log(np.clip(dens_j, 1e-12, None))
        return scores
