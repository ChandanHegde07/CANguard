"""Score normalization and hybrid combination rules for PIRD + global branch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evaluation.threshold import choose_threshold_from_val_normals

EPS = 1e-9

HYBRID_RULES = ("or", "max", "weighted")
DEFAULT_LAMBDAS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class ScoreNormalizer:
    """Freeze branch score scale using validation-normal scores only."""

    center: float
    scale: float
    provenance: str = "train_val_normal"

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.provenance != "train_val_normal":
            raise AssertionError(
                f"score normalizer provenance {self.provenance!r} != 'train_val_normal'"
            )
        return (np.asarray(scores, dtype=float) - self.center) / (self.scale + EPS)


def fit_score_normalizer(val_scores: np.ndarray) -> ScoreNormalizer:
    """Fit a z-score normalizer from a branch's validation-normal scores."""
    x = np.asarray(val_scores, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return ScoreNormalizer(center=0.0, scale=1.0)
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= EPS:
        scale = 1.0
    return ScoreNormalizer(center=float(np.mean(x)), scale=scale)


def combine_scores(
    local_z: np.ndarray,
    global_z: np.ndarray,
    *,
    rule: str,
    lam: float = 0.5,
) -> np.ndarray:
    """Combine normalized branch scores into a hybrid score.

    ``or`` is a decision-level rule (see :func:`or_predict`) and has no score;
    ``max`` and ``weighted`` return scores.
    """
    local_z = np.asarray(local_z, dtype=float)
    global_z = np.asarray(global_z, dtype=float)
    if rule == "max":
        return np.maximum(local_z, global_z)
    if rule == "weighted":
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lambda must be in [0, 1], got {lam}")
        return lam * local_z + (1.0 - lam) * global_z
    raise ValueError(f"rule must be one of {HYBRID_RULES}; 'or' has no score")


def or_predict(local_pred: np.ndarray, global_pred: np.ndarray) -> np.ndarray:
    """Decision-level OR rule."""
    return (np.asarray(local_pred).astype(int) | np.asarray(global_pred).astype(int)).astype(int)


def threshold_at_fpr(val_normal_scores: np.ndarray, fpr: float) -> float:
    """Threshold a score at a target FPR using the existing percentile rule."""
    return choose_threshold_from_val_normals(val_normal_scores, fpr)


__all__ = [
    "DEFAULT_LAMBDAS",
    "HYBRID_RULES",
    "ScoreNormalizer",
    "combine_scores",
    "fit_score_normalizer",
    "or_predict",
    "threshold_at_fpr",
]
