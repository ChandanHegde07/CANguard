
from __future__ import annotations

import numpy as np
import pandas as pd

from ..detectors.base import BaseAnomalyDetector
from .metrics import compute_metrics
from .threshold import choose_threshold_from_val_normals


def _fit_threshold_score(
    detector: BaseAnomalyDetector,
    X_fit: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fpr_target: float = 0.01,
) -> dict:
    detector.fit(X_fit)
    scores_val = detector.score_samples(X_val)
    thr = choose_threshold_from_val_normals(scores_val, fpr_target)
    scores = detector.score_samples(X_test)
    pred = (scores >= thr).astype(int)
    m = compute_metrics(y_test, pred, scores)
    m["threshold"] = thr
    m["scores"] = scores
    m["y_pred"] = pred
    return m


def permutation_importance(
    detector_factory,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    n_repeats: int = 5,
    seed: int = 0,
    fpr_target: float = 0.01,
    val_holdout_fraction: float = 0.2,
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """Permute each test feature and measure drop in ``metric`` (higher is better)."""
    train_norm = train_df[train_df["is_attack"] == 0]
    n_val = max(1, int(len(train_norm) * val_holdout_fraction))
    fit_df = train_norm.iloc[:-n_val] if n_val < len(train_norm) else train_norm
    val_df = train_norm.iloc[-n_val:] if n_val < len(train_norm) else train_norm

    X_fit = fit_df[feature_cols].fillna(0).values
    X_val = val_df[feature_cols].fillna(0).values
    X_test = test_df[feature_cols].fillna(0).values.copy()
    y_test = test_df["is_attack"].values

    base_det = detector_factory()
    base = _fit_threshold_score(base_det, X_fit, X_val, X_test, y_test, fpr_target)
    base_score = float(base[metric])

    rng = np.random.default_rng(seed)
    rows = []
    for j, col in enumerate(feature_cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            rng.shuffle(Xp[:, j])
            det = detector_factory()
            # Reuse fitted model would be faster but permutation is on test only:
            # refit not needed if we only permute test — score with base model.
            scores = base_det.score_samples(Xp)
            pred = (scores >= base["threshold"]).astype(int)
            m = compute_metrics(y_test, pred, scores)
            drops.append(base_score - float(m[metric]))
        rows.append(
            {
                "feature": col,
                "method": "permutation",
                "metric": metric,
                "baseline": base_score,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )
    out = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def leave_one_feature_out(
    detector_factory,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    fpr_target: float = 0.01,
    val_holdout_fraction: float = 0.2,
) -> pd.DataFrame:
    """Retrain without each feature; report ΔF1, ΔROC, ΔPR vs full set."""
    train_norm = train_df[train_df["is_attack"] == 0]
    n_val = max(1, int(len(train_norm) * val_holdout_fraction))
    fit_df = train_norm.iloc[:-n_val] if n_val < len(train_norm) else train_norm
    val_df = train_norm.iloc[-n_val:] if n_val < len(train_norm) else train_norm
    y_test = test_df["is_attack"].values

    def eval_cols(cols: list[str]) -> dict:
        det = detector_factory()
        return _fit_threshold_score(
            det,
            fit_df[cols].fillna(0).values,
            val_df[cols].fillna(0).values,
            test_df[cols].fillna(0).values,
            y_test,
            fpr_target,
        )

    full = eval_cols(feature_cols)
    rows = []
    for col in feature_cols:
        reduced = [c for c in feature_cols if c != col]
        m = eval_cols(reduced)
        rows.append(
            {
                "feature": col,
                "method": "lofo",
                "f1_full": full["f1"],
                "roc_auc_full": full["roc_auc"],
                "pr_auc_full": full["pr_auc"],
                "f1_without": m["f1"],
                "roc_auc_without": m["roc_auc"],
                "pr_auc_without": m["pr_auc"],
                "delta_f1": full["f1"] - m["f1"],
                "delta_roc_auc": full["roc_auc"] - m["roc_auc"],
                "delta_pr_auc": full["pr_auc"] - m["pr_auc"],
            }
        )
    out = pd.DataFrame(rows).sort_values("delta_roc_auc", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)
