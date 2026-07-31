"""Tests for evaluation metrics and threshold selection."""

from __future__ import annotations

import numpy as np

from canguard.evaluation import (
    choose_threshold_from_val_normals,
    compute_metrics,
    sweep_thresholds,
)


def test_compute_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    scores = np.where(y == 1, 5.0, 0.5)
    m = compute_metrics(y, pred, scores)
    assert m["recall"] == 1.0
    assert m["fpr"] == 0.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


def test_compute_metrics_handles_no_positive_pred():
    y = np.zeros(8)
    pred = np.zeros(8)
    scores = np.zeros(8)
    m = compute_metrics(y, pred, scores)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_threshold_targets_fpr_on_normals():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, 5_000)
    th = choose_threshold_from_val_normals(normal, fpr_target=0.01)
    fpr = (normal >= th).mean()
    assert abs(fpr - 0.01) < 0.01


def test_sweep_thresholds_ordering():
    rng = np.random.default_rng(1)
    val_norm = rng.normal(0, 1, 2000)
    test_scores = rng.normal(0, 1, 400)
    y_test = rng.integers(0, 2, 400)
    rows = sweep_thresholds(val_norm, test_scores, y_test, [0.001, 0.01, 0.05])
    assert [r["target_FPR"] for r in rows] == [0.001, 0.01, 0.05]
    # Looser target FPR -> threshold lower -> recall cannot decrease materially
    assert (
        rows[2]["recall"] >= rows[0]["recall"] or abs(rows[2]["recall"] - rows[0]["recall"]) < 1e-9
    )
