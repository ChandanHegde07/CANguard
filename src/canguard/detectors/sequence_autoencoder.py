"""Lightweight sequence reconstruction baseline (CANet / LSTM-AE inspired).

This is an **external comparison point**, not a new SOTA claim.

Design (explicit comparison basis):
  * Operates on the **same 14-d window feature matrix** (raw or residual) as
    every other detector in the CANguard harness — not raw CAN byte streams.
  * At each index i, builds a short temporal sequence of the last L feature
    vectors (zero-padded at the start), flattens it, and trains an MLP
    autoencoder to reconstruct that sequence. Anomaly score = MSE.
  * Implemented with sklearn MLPRegressor for commodity-CPU feasibility
    (no PyTorch/CUDA required). Functionally a sequence-MLP-AE stand-in for
    the class of reconstruction / deep-sequence methods cited in related work.

If a true LSTM-over-raw-frames baseline is needed later, it should be a
separate detector with a documented protocol deviation (different input).
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import BaseAnomalyDetector


class SequenceAutoencoderDetector(BaseAnomalyDetector):
    """Short-sequence MLP autoencoder on window feature vectors."""

    name = "sequence_autoencoder"

    def __init__(
        self,
        seq_len: int = 5,
        hidden_layer_sizes: tuple[int, ...] = (32, 16, 32),
        max_iter: int = 60,
        random_state: int = 0,
        max_train_samples: int = 6000,
    ) -> None:
        self.seq_len = int(seq_len)
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
            n_iter_no_change=8,
        )
        self.n_features_: int | None = None

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        L = self.seq_len
        # Pad start with first row so every index has a sequence ending at i
        pad = np.repeat(X[:1], L - 1, axis=0) if n else np.zeros((L - 1, d))
        Xp = np.vstack([pad, X])
        seqs = np.stack([Xp[i : i + L].reshape(-1) for i in range(n)], axis=0)
        return seqs

    def fit(self, X: np.ndarray) -> SequenceAutoencoderDetector:
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]
        if len(X) > self.max_train_samples:
            rng = np.random.RandomState(self.random_state)
            # keep temporal order: take a contiguous tail chunk for realism
            start = int(rng.randint(0, max(1, len(X) - self.max_train_samples + 1)))
            X = X[start : start + self.max_train_samples]
        seq = self._make_sequences(X)
        Xs = self.scaler_.fit_transform(seq)
        self.model_.fit(Xs, Xs)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        seq = self._make_sequences(X)
        Xs = self.scaler_.transform(seq)
        recon = self.model_.predict(Xs)
        return np.mean((Xs - recon) ** 2, axis=1)
