"""Tests for block bootstrap CIs."""

from __future__ import annotations

import numpy as np

from canguard.evaluation import bootstrap_metrics


def test_bootstrap_perfect_separation():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 20)
    scores = np.where(y == 1, 2.0, 0.0)
    pred = (scores >= 1.0).astype(int)
    boot = bootstrap_metrics(y, pred, scores, n_boot=50, block_size=8, seed=0)
    assert boot["f1"]["point"] == 1.0
    assert boot["f1"]["ci_low"] >= 0.9
    assert boot["roc_auc"]["point"] == 1.0
