"""Deployment-oriented evaluation metrics for PIRD.

Research metrics (F1 at a tuned operating point) are necessary but not sufficient
to judge whether a detector is deployable. This module adds the operating-point
metrics an automotive IDS is actually specified against:

* **Recall@fixed FPR** — how much attack traffic is caught at an FPR budget.
* **FPR@target recall** — the false-positive cost of reaching a safety recall.
* **False alarms per hour** — absolute alarm load on normal traffic.
* **Detection latency** — time between attack onset and first detection.

All functions are pure and operate on frozen scores, so they can be reused for
temporal, cross-capture and cross-domain validation alike.
"""

from __future__ import annotations

import numpy as np

from .latency import multi_segment_latencies
from .metrics import compute_metrics

DEFAULT_FPR_TARGETS: tuple[float, ...] = (0.001, 0.01, 0.05)
DEFAULT_RECALL_TARGET = 0.95


def threshold_at_fpr(normal_scores: np.ndarray, fpr: float) -> float:
    """Score threshold that admits ``fpr`` of the supplied normal scores."""
    normal_scores = np.asarray(normal_scores, dtype=float)
    if normal_scores.size == 0:
        return float("nan")
    return float(np.percentile(normal_scores, (1.0 - float(fpr)) * 100.0))


def threshold_at_recall(
    y_true: np.ndarray, scores: np.ndarray, recall_target: float = DEFAULT_RECALL_TARGET
) -> float:
    """Lowest score threshold that still reaches ``recall_target``.

    Ties are handled conservatively: the returned threshold is the score of the
    first ranked window at which the cumulative true-positive count meets the
    target, so ``scores >= threshold`` recovers at least the requested recall.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    need = int(np.ceil(float(recall_target) * n_pos))
    order = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(y_true[order] == 1)
    idx = int(np.searchsorted(tp, need))
    if idx >= len(scores):
        return float("-inf")
    return float(scores[order][idx])


def recall_at_fpr(
    normal_scores: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    fpr: float,
) -> dict:
    """Recall/precision/F1 when the threshold is set to admit ``fpr`` normals."""
    threshold = threshold_at_fpr(normal_scores, fpr)
    if not np.isfinite(threshold):
        return {
            "target_fpr": float(fpr),
            "threshold": float("nan"),
            "actual_fpr": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
            "f1": float("nan"),
        }
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    m = compute_metrics(y_true, y_pred, scores)
    return {
        "target_fpr": float(fpr),
        "threshold": float(threshold),
        "actual_fpr": float(m["fpr"]),
        "recall": float(m["recall"]),
        "precision": float(m["precision"]),
        "f1": float(m["f1"]),
    }


def fpr_at_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    recall_target: float = DEFAULT_RECALL_TARGET,
) -> dict:
    """FPR and operating point when the detector is forced to ``recall_target``."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    threshold = threshold_at_recall(y_true, scores, recall_target)
    if not np.isfinite(threshold):
        return {
            "recall_target": float(recall_target),
            "threshold": float(threshold),
            "recall": float("nan"),
            "fpr": float("nan"),
            "precision": float("nan"),
        }
    y_pred = (scores >= threshold).astype(int)
    m = compute_metrics(y_true, y_pred, scores)
    return {
        "recall_target": float(recall_target),
        "threshold": float(threshold),
        "recall": float(m["recall"]),
        "fpr": float(m["fpr"]),
        "precision": float(m["precision"]),
    }


def false_alarms_per_hour(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """False-positive windows normalised by observed wall-clock hours.

    The observation window is the span of window timestamps; over that span the
    detector raised ``fp`` false alarms on normal windows, so the rate is
    ``fp / hours``.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_false_alarms = int(((y_pred == 1) & (y_true == 0)).sum())
    if timestamps.size >= 2:
        duration_s = float(np.nanmax(timestamps) - np.nanmin(timestamps))
    else:
        duration_s = 0.0
    hours = duration_s / 3600.0
    fa_per_hour = n_false_alarms / hours if hours > 0 else float("nan")
    return {
        "n_false_alarms": n_false_alarms,
        "observed_hours": float(hours),
        "false_alarms_per_hour": float(fa_per_hour),
    }


def latency_report(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Detection latency aggregated over contiguous attack segments."""
    seg = multi_segment_latencies(timestamps, y_true, y_pred)
    if len(seg) == 0:
        return {
            "n_attack_segments": 0,
            "n_detected_segments": 0,
            "segment_detection_rate": float("nan"),
            "mean_delay_ms": float("nan"),
            "median_delay_ms": float("nan"),
            "max_delay_ms": float("nan"),
        }
    detected = seg["detected"].astype(bool)
    delays = seg.loc[detected, "delay_ms"].to_numpy(dtype=float)
    return {
        "n_attack_segments": int(len(seg)),
        "n_detected_segments": int(detected.sum()),
        "segment_detection_rate": float(detected.mean()),
        "mean_delay_ms": float(np.nanmean(delays)) if delays.size else float("nan"),
        "median_delay_ms": float(np.nanmedian(delays)) if delays.size else float("nan"),
        "max_delay_ms": float(np.nanmax(delays)) if delays.size else float("nan"),
    }


def deployment_report(
    normal_scores: np.ndarray,
    timestamps: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    fpr_targets: tuple[float, ...] = DEFAULT_FPR_TARGETS,
    recall_target: float = DEFAULT_RECALL_TARGET,
) -> dict:
    """Bundle Recall@FPR, FPR@recall, alarm rate and latency into one report."""
    return {
        "recall_at_fpr": [
            recall_at_fpr(normal_scores, y_true, scores, f)
            for f in fpr_targets
        ],
        "fpr_at_recall": fpr_at_recall(y_true, scores, recall_target),
        **false_alarms_per_hour(timestamps, y_true, y_pred),
        **latency_report(timestamps, y_true, y_pred),
    }


def flatten_deployment(report: dict) -> dict:
    """Flatten :func:`deployment_report` into scalar columns for CSV tables."""
    row: dict[str, float] = {}
    for entry in report.get("recall_at_fpr", []):
        f = entry["target_fpr"]
        row[f"recall_at_fpr_{f:g}"] = entry["recall"]
        row[f"actual_fpr_at_{f:g}"] = entry["actual_fpr"]
    far = report.get("fpr_at_recall", {})
    row["fpr_at_recall_target"] = far.get("recall_target", float("nan"))
    row["fpr_at_recall"] = far.get("fpr", float("nan"))
    row["recall_at_target"] = far.get("recall", float("nan"))
    row["n_false_alarms"] = report.get("n_false_alarms", float("nan"))
    row["observed_hours"] = report.get("observed_hours", float("nan"))
    row["false_alarms_per_hour"] = report.get("false_alarms_per_hour", float("nan"))
    row["n_attack_segments"] = report.get("n_attack_segments", float("nan"))
    row["n_detected_segments"] = report.get("n_detected_segments", float("nan"))
    row["segment_detection_rate"] = report.get("segment_detection_rate", float("nan"))
    row["mean_delay_ms"] = report.get("mean_delay_ms", float("nan"))
    row["median_delay_ms"] = report.get("median_delay_ms", float("nan"))
    row["max_delay_ms"] = report.get("max_delay_ms", float("nan"))
    return row


__all__ = [
    "DEFAULT_FPR_TARGETS",
    "DEFAULT_RECALL_TARGET",
    "deployment_report",
    "false_alarms_per_hour",
    "flatten_deployment",
    "fpr_at_recall",
    "latency_report",
    "recall_at_fpr",
    "threshold_at_fpr",
    "threshold_at_recall",
]
