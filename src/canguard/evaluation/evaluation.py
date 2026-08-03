"""High-level anomaly-detector evaluation orchestration."""

from __future__ import annotations

import pandas as pd

from ..detectors.base import BaseAnomalyDetector
from ..exp.resources import measure_model_size_bytes, peak_rss_mb, timed
from .metrics import compute_metrics
from .threshold import choose_threshold_from_val_normals

FPR_TARGET_DEFAULT = 0.01
VAL_HOLDOUT_FRACTION = 0.2


def train_anomaly_detector(
    detector: BaseAnomalyDetector,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    val_holdout_fraction: float = VAL_HOLDOUT_FRACTION,
    fpr_target: float = FPR_TARGET_DEFAULT,
    measure_resources: bool = True,
) -> dict:
    """Train on normal-only windows, threshold on val normals, evaluate on test."""
    train_norm = train_df[train_df["is_attack"] == 0].copy()
    n = len(train_norm)
    n_val = max(1, int(n * val_holdout_fraction))
    if n_val < n:
        train_fit = train_norm.iloc[:-n_val]
        val_norm = train_norm.iloc[-n_val:]
    else:
        train_fit = train_norm
        val_norm = train_norm

    X_fit = train_fit[feature_cols].fillna(0).values
    X_val = val_norm[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["is_attack"].values

    train_seconds = 0.0
    score_seconds = 0.0
    rss_end = peak_rss_mb() if measure_resources else 0.0

    if measure_resources:
        with timed() as t_fit:
            detector.fit(X_fit)
        train_seconds = t_fit["seconds"]
        scores_val = detector.score_samples(X_val)
        threshold = choose_threshold_from_val_normals(scores_val, fpr_target)
        detector.set_threshold(threshold)
        with timed() as t_score:
            scores_test = detector.score_samples(X_test)
        score_seconds = t_score["seconds"]
        rss_end = t_score.get("rss_mb_end", peak_rss_mb())
    else:
        detector.fit(X_fit)
        scores_val = detector.score_samples(X_val)
        threshold = choose_threshold_from_val_normals(scores_val, fpr_target)
        detector.set_threshold(threshold)
        scores_test = detector.score_samples(X_test)

    y_pred = (scores_test >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, scores_test)

    model_bytes = 0
    if measure_resources:
        try:
            model_bytes = measure_model_size_bytes(detector)
        except Exception:
            model_bytes = 0

    metrics.update(
        {
            "threshold": threshold,
            "scores_test": scores_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "model": detector,
            "thresh": threshold,
            "n_train_normals": len(train_fit),
            "train_seconds": train_seconds,
            "score_seconds": score_seconds,
            "runtime_seconds": train_seconds + score_seconds,
            "peak_rss_mb": rss_end,
            "model_bytes": model_bytes,
        }
    )
    return metrics


def cross_attack_evaluate(
    detector: BaseAnomalyDetector,
    source_train_normals_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    residual_feature_cols: list[str],
    val_holdout_fraction: float = VAL_HOLDOUT_FRACTION,
    fpr_target: float = FPR_TARGET_DEFAULT,
) -> dict[str, float]:
    """Train on source normals, threshold on source holdout, test target."""
    source = source_train_normals_df.reset_index(drop=True)
    n = len(source)
    if n < 10:
        return {
            "recall": float("nan"),
            "fpr": float("nan"),
            "f1": float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    n_val = max(1, int(n * val_holdout_fraction))
    source_fit = source.iloc[:-n_val] if n_val < n else source
    source_val = source.iloc[-n_val:] if n_val < n else source

    detector.fit(source_fit[residual_feature_cols].fillna(0).values)
    scores_val = detector.score_samples(source_val[residual_feature_cols].fillna(0).values)
    threshold = choose_threshold_from_val_normals(scores_val, fpr_target)

    X_test = target_test_df[residual_feature_cols].fillna(0).values
    y_test = target_test_df["is_attack"].values
    scores_test = detector.score_samples(X_test)
    y_pred = (scores_test >= threshold).astype(int)

    m = compute_metrics(y_test, y_pred, scores_test)
    return {
        "recall": m["recall"],
        "fpr": m["fpr"],
        "f1": m["f1"],
        "tp": m["tp"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tn": m["tn"],
        "threshold": threshold,
    }
