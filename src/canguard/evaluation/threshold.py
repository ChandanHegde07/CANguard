
from __future__ import annotations

import numpy as np


def choose_threshold_from_val_normals(
    val_normal_scores: np.ndarray, fpr_target: float = 0.01
) -> float:

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
