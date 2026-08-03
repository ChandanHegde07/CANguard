"""Paired statistical tests for representation comparisons.

McNemar's test compares paired binary predictions on the same windows
(residual vs raw, or detector A vs B). Block bootstrap remains preferred for
metric CIs under temporal dependence.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar test on paired predictions (continuity-corrected chi-square).

    Contests whether the two classifiers disagree in a biased way.
    Only windows where exactly one of (pred_a, pred_b) is correct contribute.

    Appropriate when:
      * same test instances
      * binary decisions
      * interest is disagreement, not independent samples
    """
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    # b: A wrong, B right; c: A right, B wrong
    b = int((~correct_a & correct_b).sum())
    c = int((correct_a & ~correct_b).sum())
    # continuity-corrected McNemar
    if b + c == 0:
        return {
            "b_a_wrong_b_right": b,
            "c_a_right_b_wrong": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "n_discordant": 0,
        }
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(1.0 - stats.chi2.cdf(stat, df=1))
    return {
        "b_a_wrong_b_right": b,
        "c_a_right_b_wrong": c,
        "statistic": float(stat),
        "p_value": p,
        "n_discordant": b + c,
    }


def paired_bootstrap_delta(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 400,
    block_size: int = 50,
    seed: int = 0,
    metric: str = "f1",
) -> dict:
    """Block-bootstrap CI for metric(B) - metric(A) on paired indices."""
    from .metrics import compute_metrics

    y_true = np.asarray(y_true)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    def _metric(yt, pa, sa):
        return float(compute_metrics(yt, pa, sa)[metric])

    point = _metric(y_true, pred_b, scores_b) - _metric(y_true, pred_a, scores_a)
    samples = []
    block_size = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / block_size))
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])[:n]
        if len(idx) < n:
            idx = np.concatenate([idx, rng.integers(0, n, size=n - len(idx))])
        d = _metric(y_true[idx], pred_b[idx], scores_b[idx]) - _metric(
            y_true[idx], pred_a[idx], scores_a[idx]
        )
        samples.append(d)
    arr = np.asarray(samples)
    return {
        "metric": metric,
        "delta_point": float(point),
        "delta_mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_boot": n_boot,
        "block_size": block_size,
    }
