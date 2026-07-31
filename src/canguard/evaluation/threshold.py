"""Threshold selection for anomaly scores."""

from __future__ import annotations

import numpy as np


def choose_threshold_from_val_normals(
    val_normal_scores: np.ndarray, fpr_target: float = 0.01
) -> float:
    """Choose a score threshold targeting a given FPR on validation normals.

    The threshold is the ``(1 - fpr_target)``-th percentile of normal scores,
    so approximately ``fpr_target`` of normals score above it (flagged).

    Parameters
    ----------
    val_normal_scores : np.ndarray
        Anomaly scores on validation **normal** samples only.
    fpr_target : float
        Desired false-positive rate on normals (default 0.01 ~ 1%).

    Returns
    -------
    float
        Score threshold to use for hard predictions.
    """
    val_normal_scores = np.asarray(val_normal_scores)
    if val_normal_scores.size == 0:
        return 0.0
    return float(np.percentile(val_normal_scores, (1 - fpr_target) * 100))


def sweep_thresholds(
    val_normal_scores: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    fpr_targets: list[float],
) -> list[dict[str, float]]:
    """Evaluate a set of FPR targets and report test metrics at each.

    For each ``fpr_target``, selects the percentile-based threshold on
    ``val_normal_scores``, then reports actual FPR / recall / precision / F1 on
    the test set (using :func:`compute_metrics`).

    Parameters
    ----------
    val_normal_scores : np.ndarray
        Scores on validation normals (for threshold selection).
    test_scores : np.ndarray
        Scores on the test set.
    y_test : np.ndarray
        Ground-truth labels on the test set.
    fpr_targets : list[float]
        Target FPRs (e.g. [0.001, 0.01, 0.05]).

    Returns
    -------
    list[dict[str, float]]
        One dict per target with keys target_FPR, actual_FPR, precision,
        recall, f1 (and tp/fp/fn/tn).
    """
    from .metrics import compute_metrics

    y_test = np.asarray(y_test)
    test_scores = np.asarray(test_scores)
    rows = []
    for target in fpr_targets:
        th = choose_threshold_from_val_normals(val_normal_scores, target)
        y_pred = (test_scores >= th).astype(int)
        m = compute_metrics(y_test, y_pred, test_scores)
        rows.append(
            {
                "target_FPR": float(target),
                "actual_FPR": m["fpr"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tn": m["tn"],
            }
        )
    return rows
