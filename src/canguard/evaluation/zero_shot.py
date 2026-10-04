"""Zero-shot evaluation for Phase 4.

The actual zero-shot operating point uses a threshold selected **only** from
source validation normals. ``tpr_at_{1,5}pct_fpr`` are oracle diagnostics whose
thresholds are selected on target normals; they characterise score separability
and are never presented as the zero-shot result. Actual target FPR is reported
for both.
"""

from __future__ import annotations

import numpy as np

from .metrics import compute_metrics


def aggregate_frame_scores(
    window_scores: np.ndarray,
    frame_rows: np.ndarray,
    n_frames: int,
    position_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Average per-window scores onto frames.

    ``window_scores`` [M, K] corresponds to frame positions
    ``frame_rows[:, position_offset:position_offset+K]``. Returns
    ``(frame_score [n_frames], frame_count [n_frames])``.
    """
    window_scores = np.asarray(window_scores, dtype=float)
    frame_rows = np.asarray(frame_rows)
    k = window_scores.shape[1]
    idx = frame_rows[:, position_offset : position_offset + k]
    scores = np.zeros(n_frames, dtype=float)
    counts = np.zeros(n_frames, dtype=float)
    mask = np.ones_like(idx, dtype=bool)
    np.add.at(scores, idx[mask], window_scores[mask])
    np.add.at(counts, idx[mask], 1.0)
    out = np.divide(scores, counts, out=np.full_like(scores, np.nan), where=counts > 0)
    return out, counts


def threshold_from_normals(normal_scores: np.ndarray, fpr: float) -> float:
    normal_scores = np.asarray(normal_scores, dtype=float)
    normal_scores = normal_scores[np.isfinite(normal_scores)]
    if normal_scores.size == 0:
        return float("inf")
    return float(np.percentile(normal_scores, (1.0 - fpr) * 100.0))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, normals: np.ndarray, fpr: float) -> dict:
    thr = threshold_from_normals(normals, fpr)
    y = np.asarray(labels).astype(int)
    pred = (np.asarray(scores) >= thr).astype(int)
    pos = int(y.sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    neg = int((y == 0).sum())
    return {
        "threshold": thr,
        "tpr": (tp / pos) if pos else float("nan"),
        "actual_fpr": (fp / neg) if neg else float("nan"),
    }


def evaluate_zero_shot_scores(
    source_normal_scores: np.ndarray,
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    fpr_target: float = 0.01,
    nominal_fprs: tuple[float, ...] = (0.01, 0.05),
) -> dict:
    source_normal_scores = np.asarray(source_normal_scores, dtype=float)
    source_normal_scores = source_normal_scores[np.isfinite(source_normal_scores)]
    target_scores = np.asarray(target_scores, dtype=float)
    target_labels = np.asarray(target_labels).astype(int)
    finite = np.isfinite(target_scores)
    target_scores = target_scores[finite]
    target_labels = target_labels[finite]
    target_normals = target_scores[target_labels == 0]

    threshold = threshold_from_normals(source_normal_scores, fpr_target)
    pred = (target_scores >= threshold).astype(int)
    m = compute_metrics(target_labels, pred, target_scores)

    source_val_fpr = (
        float((source_normal_scores >= threshold).mean())
        if source_normal_scores.size
        else float("nan")
    )
    out = {
        "source_threshold": threshold,
        "source_validation_fpr": source_val_fpr,
        "n_target": int(len(target_scores)),
        "n_target_normal": int((target_labels == 0).sum()),
        "n_target_attack": int((target_labels == 1).sum()),
        "target_actual_fpr": m["fpr"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tn": m["tn"],
    }
    for f in nominal_fprs:
        diag = tpr_at_fpr(target_scores, target_labels, target_normals, f)
        out[f"tpr_at_{f:g}_fpr"] = diag["tpr"]
        out[f"actual_fpr_at_{f:g}_fpr"] = diag["actual_fpr"]
    return out


def sustained_detection_delay(
    times: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    n_of_m: tuple[int, int] = (3, 5),
) -> dict:
    """Detection delay for attack episodes using an N-of-M sustained criterion.

    Episodes are maximal runs of consecutive attack frames. Delay is the time from
    episode start to the first window of M frames containing >= N alarms.
    """
    times = np.asarray(times, dtype=float)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    alarms = (scores >= threshold).astype(int)
    n, m = n_of_m
    delays = []
    detected = 0
    episodes = []
    i = 0
    while i < len(labels):
        if labels[i] == 1:
            j = i
            while j + 1 < len(labels) and labels[j + 1] == 1:
                j += 1
            episodes.append((i, j))
            i = j + 1
        else:
            i += 1
    for a, b in episodes:
        t0 = times[a]
        found = None
        for t in range(a, min(b + 1, len(labels))):
            lo = max(a, t - m + 1)
            if alarms[lo : t + 1].sum() >= n:
                found = t
                break
        if found is not None:
            detected += 1
            delays.append(float(times[found] - t0))
    return {
        "n_episodes": len(episodes),
        "n_detected": detected,
        "detection_rate": (detected / len(episodes)) if episodes else float("nan"),
        "median_delay_s": float(np.median(delays)) if delays else float("nan"),
        "mean_delay_s": float(np.mean(delays)) if delays else float("nan"),
    }


__all__ = [
    "aggregate_frame_scores",
    "evaluate_zero_shot_scores",
    "sustained_detection_delay",
    "threshold_from_normals",
    "tpr_at_fpr",
]
