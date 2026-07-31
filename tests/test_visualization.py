"""Smoke tests for visualization helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canguard.visualization import (
    plot_ablation_heatmap,
    plot_cross_attack_matrix,
    plot_roc_pr,
    plot_score_distribution,
    plot_score_distribution_grid,
    plot_score_timeline,
    plot_threshold_sweep,
)


def _dummy_result(n=200, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 1, n)
    y = rng.integers(0, 2, n)
    return {"scores_test": scores, "y_test": y, "threshold": 1.5}


def test_score_distribution_returns_axes():
    r = _dummy_result()
    ax = plot_score_distribution(r["scores_test"], r["y_test"], r["threshold"], title="t")
    assert ax is not None
    plt.close("all")


def test_score_distribution_grid():
    results = {"DoS": _dummy_result(), "RPM": _dummy_result(), "gear": _dummy_result()}
    fig = plot_score_distribution_grid(results, ["DoS", "RPM", "gear"], ncols=2)
    assert fig is not None
    plt.close("all")


def test_roc_pr():
    y = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.7])
    fig = plot_roc_pr(y, scores, fpr_at_op=0.2)
    assert fig is not None
    plt.close("all")


def test_threshold_sweep():
    rows = [
        {"target_FPR": 0.001, "recall": 0.4, "actual_FPR": 0.001},
        {"target_FPR": 0.01, "recall": 0.7, "actual_FPR": 0.011},
        {"target_FPR": 0.05, "recall": 0.9, "actual_FPR": 0.049},
    ]
    ax = plot_threshold_sweep(rows, title="sweep")
    assert ax is not None
    plt.close("all")


def test_cross_attack_matrix():
    rec = pd.DataFrame({"A": [1.0, 0.5], "B": [0.4, 1.0]}, index=["A", "B"])
    fpr = pd.DataFrame({"A": [0.01, 0.5], "B": [0.2, 0.01]}, index=["A", "B"])
    fig = plot_cross_attack_matrix(rec, fpr)
    assert fig is not None
    plt.close("all")


def test_ablation_heatmap():
    df = pd.DataFrame({"RPM": [0.9, 0.8], "DoS": [0.7, 0.6]}, index=["A", "B"])
    fig = plot_ablation_heatmap(df)
    assert fig is not None
    plt.close("all")


def test_timeline():
    scores = np.random.default_rng(0).normal(0, 1, 100)
    y = np.zeros(100)
    y[40:60] = 1
    fig = plot_score_timeline(scores, y, threshold=1.5)
    assert fig is not None
    plt.close("all")
