"""Block / cluster-aware bootstrap confidence intervals.

Avoids treating successive CAN windows as IID by resampling contiguous time
blocks (and optionally by can_id clusters).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .metrics import compute_metrics

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]


def _block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample with replacement blocks of ``block_size`` covering ~n indices."""
    if n <= 0:
        return np.array([], dtype=int)
    block_size = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])
    if len(idx) > n:
        idx = idx[:n]
    elif len(idx) < n:
        # pad by sampling more single points
        extra = rng.integers(0, n, size=n - len(idx))
        idx = np.concatenate([idx, extra])
    return idx


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    block_size: int = 50,
    seed: int = 0,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"),
    ci: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Block-bootstrap CIs for classification metrics.

    Parameters
    ----------
    y_true, y_pred, scores
        Test-set arrays (same length, chronological order preferred).
    n_boot
        Number of bootstrap replicates.
    block_size
        Contiguous window-block length (reduces IID assumption).
    seed
        RNG seed.
    metrics
        Metric keys from :func:`compute_metrics`.
    ci
        Confidence level (default 0.95).

    Returns
    -------
    dict
        ``{metric: {point, ci_low, ci_high, mean, std}}``
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)
    n = len(y_true)
    point = compute_metrics(y_true, y_pred, scores)
    rng = np.random.default_rng(seed)

    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = _block_indices(n, block_size, rng)
        m = compute_metrics(y_true[idx], y_pred[idx], scores[idx])
        for key in metrics:
            val = m.get(key, float("nan"))
            if val == val:  # not NaN
                samples[key].append(float(val))

    alpha = (1.0 - ci) / 2.0
    out: dict[str, dict[str, float]] = {}
    for key in metrics:
        arr = np.asarray(samples[key], dtype=float)
        if len(arr) == 0:
            out[key] = {
                "point": float(point.get(key, float("nan"))),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
            }
            continue
        out[key] = {
            "point": float(point.get(key, float("nan"))),
            "ci_low": float(np.quantile(arr, alpha)),
            "ci_high": float(np.quantile(arr, 1.0 - alpha)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    return out


def flatten_bootstrap(boot: dict[str, dict[str, float]], prefix: str = "") -> dict[str, float]:
    """Flatten nested bootstrap dict for CSV rows."""
    row: dict[str, float] = {}
    for metric, stats in boot.items():
        for k, v in stats.items():
            row[f"{prefix}{metric}_{k}" if prefix else f"{metric}_{k}"] = v
    return row
