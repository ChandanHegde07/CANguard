"""Source→target transfer core for PIRD (Phase 3).

Fits a PIRD model *entirely on source normal traffic* and applies it, frozen, to
a target domain. The target never participates in per-ID statistics, Isolation
Forest fitting, or threshold selection.

Unseen target CAN IDs are explicitly identified. The primary transfer result is
computed on **known-ID** windows only; the existing global-fallback path is
reported separately as a diagnostic and is never mixed into the primary result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..detectors.base import BaseAnomalyDetector
from ..features.per_id import GlobalStats, PerIdStats, fit_per_id_stats, transform_residuals
from .threshold import choose_threshold_from_val_normals
from .transfer_metrics import (
    canonicalize_frame_ids,
    classify_window_ids,
    id_overlap,
    per_attack_metrics,
    recall_at_fpr,
    subset_metrics,
)


@dataclass
class PirdSourceModel:
    """Frozen PIRD source model: per-ID stats + detector + threshold."""

    stats: PerIdStats
    global_stats: GlobalStats
    feature_cols: list[str]
    model: BaseAnomalyDetector
    threshold: float
    val_normal_scores: np.ndarray
    n_calib: int = 0
    n_if: int = 0
    n_val: int = 0
    provenance: str = "source_normal"
    meta: dict = field(default_factory=dict)

    @property
    def res_cols(self) -> list[str]:
        return [c + "_res" for c in self.feature_cols]

    @property
    def known_ids(self) -> set[str]:
        return set(self.stats.keys())

    def residualize(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.provenance != "source_normal":
            raise AssertionError("source model provenance corrupted")
        return transform_residuals(df, self.stats, self.global_stats, self.feature_cols)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        res = self.residualize(df)
        return self.model.score_samples(res[self.res_cols].fillna(0).values)


REQUIRED_DOMAIN_COLUMNS = ("can_id", "timestamp", "is_attack")


def validate_domain_schema(
    df: pd.DataFrame,
    feature_cols: list[str],
    required: tuple[str, ...] = REQUIRED_DOMAIN_COLUMNS,
) -> None:
    """Raise ``ValueError`` if a domain table lacks required columns/features.

    Source and target must share one behavioural feature schema for a transfer
    to be meaningful; an incompatible schema is a plumbing error, not something
    to silently patch by dropping features.
    """
    missing = [c for c in (*required, *feature_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"domain schema missing columns: {missing}")


def split_captures(
    capture_names,
    target_frac: float = 0.3,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Deterministically partition captures into disjoint source/target sets.

    Capture-level (not frame-level) separation prevents leakage through
    overlapping or temporally adjacent windows within the same capture.
    """
    names = sorted({str(c) for c in capture_names})
    if len(names) < 2:
        raise ValueError("need at least 2 captures for a capture-level split")
    if not 0.0 < target_frac < 1.0:
        raise ValueError("target_frac must be in (0, 1)")
    rng = np.random.default_rng(int(seed))
    perm = [str(x) for x in rng.permutation(np.array(names, dtype=object))]
    n_target = max(1, min(len(names) - 1, int(round(target_frac * len(names)))))
    return {"source": sorted(perm[n_target:]), "target": sorted(perm[:n_target])}


def split_vehicles(
    vehicle_names,
    source_vehicles,
    target_vehicles,
) -> dict[str, list[str]]:
    """Validate and normalise a cross-vehicle source/target partition.

    Vehicles are the finest-grained domain separation available, so the two
    sides must be non-empty and disjoint.
    """
    names = {str(v) for v in vehicle_names}
    src = {str(v) for v in source_vehicles}
    tgt = {str(v) for v in target_vehicles}
    if not src or not tgt:
        raise ValueError("both source_vehicles and target_vehicles must be non-empty")
    unknown = (src | tgt) - names
    if unknown:
        raise ValueError(f"unknown vehicle(s): {sorted(unknown)}")
    if src & tgt:
        raise ValueError(f"source and target vehicles overlap: {sorted(src & tgt)}")
    return {"source": sorted(src), "target": sorted(tgt)}


def vehicle_domains(df: pd.DataFrame, vehicle_col: str = "vehicle") -> dict[str, pd.DataFrame]:
    """Split a frame/window table by vehicle identity."""
    if vehicle_col not in df.columns:
        raise ValueError(f"table has no {vehicle_col!r} column")
    return {str(v): g.reset_index(drop=True) for v, g in df.groupby(vehicle_col)}


def fit_pird_source(
    calib_normals: pd.DataFrame,
    if_normals: pd.DataFrame,
    val_normals: pd.DataFrame,
    feature_cols: list[str],
    detector: BaseAnomalyDetector,
    fpr_target: float = 0.01,
) -> PirdSourceModel:
    """Fit PIRD on source normals only (stats→calib, IF→fit, threshold→val).

    CAN IDs are canonicalised first so source and target share one ID space.
    """
    for _name, _df in (("calib", calib_normals), ("if_fit", if_normals), ("val", val_normals)):
        validate_domain_schema(_df, feature_cols)
    calib_normals = calib_normals.copy()
    if_normals = if_normals.copy()
    val_normals = val_normals.copy()
    calib_normals["can_id"] = canonicalize_frame_ids(calib_normals["can_id"])
    if_normals["can_id"] = canonicalize_frame_ids(if_normals["can_id"])
    val_normals["can_id"] = canonicalize_frame_ids(val_normals["can_id"])
    stats, gstats = fit_per_id_stats(calib_normals, feature_cols)
    res_cols = [c + "_res" for c in feature_cols]
    res_if = transform_residuals(if_normals, stats, gstats, feature_cols)
    detector.fit(res_if[res_cols].fillna(0).values)
    res_val = transform_residuals(val_normals, stats, gstats, feature_cols)
    val_scores = detector.score_samples(res_val[res_cols].fillna(0).values)
    threshold = choose_threshold_from_val_normals(val_scores, fpr_target)
    detector.set_threshold(threshold)
    return PirdSourceModel(
        stats=stats,
        global_stats=gstats,
        feature_cols=list(feature_cols),
        model=detector,
        threshold=float(threshold),
        val_normal_scores=np.asarray(val_scores, dtype=float),
        n_calib=len(calib_normals),
        n_if=len(if_normals),
        n_val=len(val_normals),
    )


def evaluate_transfer(
    source: PirdSourceModel,
    target_raw: pd.DataFrame,
    feature_cols: list[str],
    fpr_targets: tuple[float, ...] = (0.001, 0.01, 0.05, 0.10),
    label_col: str = "attack_type",
) -> dict:
    """Zero-shot evaluation of a frozen source model on a target domain.

    Returns a result dict with primary (known-ID) metrics, unseen-ID/global
    fallback diagnostics, ID overlap, and source-selected fixed-FPR diagnostics.
    """
    validate_domain_schema(target_raw, feature_cols)
    target = target_raw.copy()
    target["can_id"] = canonicalize_frame_ids(target["can_id"])
    scores = source.score(target)
    y = target["is_attack"].to_numpy(dtype=int)
    known_mask = classify_window_ids(target["can_id"].to_numpy(), source.known_ids)
    pred = (scores >= source.threshold).astype(int)

    target_frame_counts = target["can_id"].value_counts().to_dict()
    overlap = id_overlap(source.known_ids, set(target["can_id"].unique()), target_frame_counts)

    result: dict = {
        "n_target_frames": int(len(target)),
        "n_target_ids": int(target["can_id"].nunique()),
        "n_known_id_windows": int(known_mask.sum()),
        "n_unseen_id_windows": int((~known_mask).sum()),
        "target_unseen_id_window_fraction": (
            float((~known_mask).mean()) if len(target) else float("nan")
        ),
        "target_unseen_id_frame_fraction": (
            float((~known_mask).mean()) if len(target) else float("nan")
        ),
        "source_threshold": source.threshold,
        "source_validation_fpr": float((source.val_normal_scores >= source.threshold).mean())
        if source.val_normal_scores.size else float("nan"),
        **overlap,
    }

    # Primary: known-ID target windows only.
    if known_mask.any():
        m = subset_metrics(y[known_mask], pred[known_mask], scores[known_mask])
        result.update({
            "primary_precision": m["precision"], "primary_recall": m["recall"],
            "primary_f1": m["f1"], "primary_roc_auc": m["roc_auc"],
            "primary_pr_auc": m["pr_auc"], "target_actual_fpr": m["fpr"],
            "primary_tp": m["tp"], "primary_fp": m["fp"],
            "primary_fn": m["fn"], "primary_tn": m["tn"],
        })
        result["per_attack"] = per_attack_metrics(target[known_mask], pred[known_mask],
                                                  scores[known_mask], label_col).to_dict("records")
    else:
        result.update({
            "primary_precision": float("nan"), "primary_recall": float("nan"),
            "primary_f1": float("nan"), "primary_roc_auc": float("nan"),
            "primary_pr_auc": float("nan"), "target_actual_fpr": float("nan"),
        })
        result["per_attack"] = []

    # Diagnostic only: unseen-ID windows via the existing global fallback.
    if (~known_mask).any():
        mf = subset_metrics(y[~known_mask], pred[~known_mask], scores[~known_mask])
        result.update({
            "fallback_precision": mf["precision"], "fallback_recall": mf["recall"],
            "fallback_f1": mf["f1"], "fallback_roc_auc": mf["roc_auc"],
            "fallback_pr_auc": mf["pr_auc"], "fallback_actual_fpr": mf["fpr"],
        })
    else:
        result.update({k: float("nan") for k in (
            "fallback_precision", "fallback_recall", "fallback_f1",
            "fallback_roc_auc", "fallback_pr_auc", "fallback_actual_fpr")})

    # Source-selected fixed-FPR diagnostics (known-ID target windows).
    diag = []
    if known_mask.any():
        for fpr in fpr_targets:
            row = recall_at_fpr(source.val_normal_scores, y[known_mask], scores[known_mask], fpr)
            row["operating_point"] = "source_selected"
            diag.append(row)
    result["fixed_fpr"] = diag
    return result


__all__ = [
    "PirdSourceModel",
    "evaluate_transfer",
    "fit_pird_source",
    "split_captures",
    "split_vehicles",
    "validate_domain_schema",
    "vehicle_domains",
]
