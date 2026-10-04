"""Transfer metrics and CAN-ID overlap utilities for Phase 3.

All metrics operate on frozen source→target scores. No target statistic enters
fitting; these helpers are evaluation-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def canonical_can_id(value) -> str:
    """Canonical hex CAN-ID string (lowercase, no ``0x``, no leading zeros).

    Needed because HCRL stores zero-padded IDs (``0316``) while ROAD stores bare
    hex (``316`` / ``6e0``). This is a *representation* normalisation only.
    """
    s = str(value).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    try:
        return format(int(s, 16), "x")
    except (ValueError, TypeError):
        return s


def canonicalize_frame_ids(series: pd.Series) -> pd.Series:
    return series.map(canonical_can_id)


def id_overlap(
    source_ids: set[int] | set[str],
    target_ids: set[int] | set[str],
    target_frame_counts: dict | None = None,
) -> dict:
    """ID-count and (optional) frame-weighted overlap of target IDs.

    IDs must already be canonicalised to a common representation.
    """
    source_ids = set(source_ids)
    target_ids = set(target_ids)
    shared = source_ids & target_ids
    out = {
        "n_source_ids": len(source_ids),
        "n_target_ids": len(target_ids),
        "n_shared_ids": len(shared),
        "n_source_only_ids": len(source_ids - target_ids),
        "n_target_only_ids": len(target_ids - source_ids),
        "id_count_overlap": len(shared) / len(target_ids) if target_ids else float("nan"),
    }
    if target_frame_counts:
        total = sum(target_frame_counts.values())
        shared_frames = sum(v for k, v in target_frame_counts.items() if k in shared)
        out["frame_weighted_overlap"] = shared_frames / total if total else float("nan")
    else:
        out["frame_weighted_overlap"] = float("nan")
    return out


def classify_window_ids(target_can_ids, known_ids: set) -> np.ndarray:
    """Return a boolean mask: True where the window's ID is known to the source."""
    return np.array([cid in known_ids for cid in target_can_ids], dtype=bool)


def recall_at_fpr(normal_scores: np.ndarray, y_true: np.ndarray, scores: np.ndarray,
                  fpr: float) -> dict:
    """Diagnostic threshold chosen on *normal* scores to hit ``fpr``, then recall."""
    normal_scores = np.asarray(normal_scores, dtype=float)
    if normal_scores.size == 0:
        return {"target_fpr": fpr, "threshold": float("nan"), "actual_fpr": float("nan"),
                "recall": float("nan"), "precision": float("nan")}
    threshold = float(np.percentile(normal_scores, (1.0 - fpr) * 100.0))
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    m = compute_metrics(y_true, y_pred, scores)
    return {
        "target_fpr": float(fpr),
        "threshold": threshold,
        "actual_fpr": m["fpr"],
        "recall": m["recall"],
        "precision": m["precision"],
        "f1": m["f1"],
    }


def subset_metrics(y_true, y_pred, scores) -> dict:
    """Standard metrics on a subset (e.g. known-ID target windows)."""
    return compute_metrics(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int),
                           np.asarray(scores, dtype=float))


def per_attack_metrics(target_df: pd.DataFrame, y_pred: np.ndarray, scores: np.ndarray,
                       label_col: str = "attack_type") -> pd.DataFrame:
    """Attack-specific precision/recall/F1 for a target domain."""
    rows = []
    if label_col not in target_df.columns:
        return pd.DataFrame()
    y = target_df["is_attack"].to_numpy(dtype=int)
    y_pred = np.asarray(y_pred).astype(int)
    for atype in sorted(target_df[label_col].astype(str).unique()):
        mask = target_df[label_col].astype(str).to_numpy() == atype
        if mask.sum() == 0:
            continue
        m = compute_metrics(y[mask], y_pred[mask], np.asarray(scores)[mask])
        rows.append({
            "attack": atype, "attack_window_count": int(mask.sum()),
            "attack_frame_count": int((y[mask] == 1).sum()),
            "attack_recall": m["recall"], "attack_precision": m["precision"],
            "attack_f1": m["f1"], "attack_actual_fpr": m["fpr"],
        })
    return pd.DataFrame(rows)


__all__ = [
    "canonical_can_id",
    "canonicalize_frame_ids",
    "classify_window_ids",
    "id_overlap",
    "per_attack_metrics",
    "recall_at_fpr",
    "subset_metrics",
]
