"""High-level anomaly-detector evaluation orchestration.

Ports the fit-on-normals / threshold-on-validation / evaluate-on-test protocol
from ``pird_hcrl.ipynb`` into a single reusable function, plus cross-attack
comparison. The returned result dict also stores the test scores, labels, and
threshold so visualization code never needs to refit a model.
"""

from __future__ import annotations

import pandas as pd

from ..detectors.base import BaseAnomalyDetector
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
) -> dict:
    """Train on normal-only windows, threshold on val normals, evaluate on test.

    Reproduces the ``pird_hcrl.ipynb`` protocol:
    1. Take the ``train_df`` rows with ``is_attack == 0``.
    2. Hold out the last ``val_holdout_fraction`` of those normals as validation.
    3. Fit ``detector`` on the remaining normals.
    4. Score validation normals, pick the threshold at ``fpr_target`` FPR.
    5. Score the full test set, threshold, and compute metrics.

    Parameters
    ----------
    detector : BaseAnomalyDetector
        An (unfit) detector instance.
    train_df : pd.DataFrame
        Training window table (contains ``is_attack`` and ``feature_cols``).
    test_df : pd.DataFrame
        Test window table.
    feature_cols : list[str]
        Feature columns used for fitting/scoring (e.g. residual columns).
    val_holdout_fraction : float
        Fraction of train normals held out for threshold selection.
    fpr_target : float
        Target false-positive rate on val normals.

    Returns
    -------
    dict
        Keys: fpr, precision, recall, f1, roc_auc, pr_auc, threshold,
        scores_test, y_test, model.
    """
    train_norm = train_df[train_df["is_attack"] == 0].copy()
    n = len(train_norm)
    n_val = max(1, int(n * val_holdout_fraction))
    if n_val < n:
        train_fit = train_norm.iloc[:-n_val]
        val_norm = train_norm.iloc[-n_val:]
    else:
        train_fit = train_norm
        val_norm = train_norm

    detector.fit(train_fit[feature_cols].fillna(0).values)

    scores_val = detector.score_samples(val_norm[feature_cols].fillna(0).values)
    threshold = choose_threshold_from_val_normals(scores_val, fpr_target)

    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["is_attack"].values
    scores_test = detector.score_samples(X_test)
    y_pred = (scores_test >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, scores_test)
    metrics.update(
        {
            "threshold": threshold,
            "scores_test": scores_test,
            "y_test": y_test,
            "model": detector,
            "thresh": threshold,
            "n_train_normals": len(train_fit),
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
    """Train on source normals, threshold on source normals' holdout, test target.

    Port of the v1 cross-attack protocol. Single residualization is expected
    BEFORE calling this function; it never residualizes internally. This
    function owns only the fit/threshold/score step.

    Returns
    -------
    dict[str, float]
        recall, fpr, f1 keys (plus tp/fp/fn/tn, threshold).
    """
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
