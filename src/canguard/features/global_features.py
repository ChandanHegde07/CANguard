"""Global / cross-ID behavioral features for CAN bus anomaly detection.

Motivation
----------
PIRD models each CAN ID independently (local, per-ID behaviour). Some attacks,
however, act on the **population/global** structure of the bus: flooding one ID,
injecting new IDs, changing ID frequencies, or changing the relationship between
IDs. This module builds a complementary, causal, window-causal representation of
that global structure.

Design rules
------------
* **Causal**: the feature vector at row ``t`` depends only on messages at rows
  ``<= t`` (a trailing window of ``window_frames`` emissions). No future data.
* **Streamable**: every feature is a rolling count / rolling moment over a
  bounded deque, except those that reference a frozen calibration set
  (``new_id_count``, ``freq_l1_to_calib``, ...), which require historical
  calibration but still never look forward in time.
* **No per-ID residualization**: global features are population-level and are
  normalized globally (robust or z-score), not per ID.
* **Independent of the adaptive attack**: the feature families below were defined
  from CAN bus semantics, not from Phase 2 attack outcomes.

The feature table is aligned 1:1 with the per-ID window table produced by
``canguard.features.FeaturePipeline`` so that the frozen temporal split,
detectors and thresholds can be reused unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-9

# ---------------------------------------------------------------------------
# Feature definitions (families are used for the ablation study)
# ---------------------------------------------------------------------------
BUS_FEATURES: list[str] = [
    "g_n_frames",
    "g_duration_s",
    "g_frame_rate",
    "g_iat_mean",
    "g_iat_std",
    "g_iat_cv",
    "g_iat_min",
    "g_iat_max",
    "g_iat_median",
    "g_burstiness",
    "g_dlc_mean",
    "g_dlc_std",
    "g_bus_bytes_per_sec_proxy",
    "g_payload_bus_mean",
]

ID_POPULATION_FEATURES: list[str] = [
    "g_n_unique_ids",
    "g_new_id_count",
    "g_id_entropy",
    "g_id_entropy_norm",
    "g_id_concentration",
    "g_dominant_id_frac",
    "g_top3_id_frac",
    "g_n_ids_above_5pct",
    "g_active_id_ratio",
]

CROSS_ID_FEATURES: list[str] = [
    "g_id_count_cv",
    "g_freq_l1_to_calib",
    "g_id_dist_cosine_to_calib",
    "g_payload_std_across_ids",
    "g_iat_std_across_ids",
]

GLOBAL_FEATURE_FAMILIES: dict[str, list[str]] = {
    "bus": BUS_FEATURES,
    "id_population": ID_POPULATION_FEATURES,
    "cross_id": CROSS_ID_FEATURES,
}

# Pre-defined ablation feature sets (A-E in the Phase 2 plan).
GLOBAL_ABLATIONS: dict[str, list[str]] = {
    "A_bus": BUS_FEATURES,
    "B_id_population": ID_POPULATION_FEATURES,
    "C_cross_id": CROSS_ID_FEATURES,
    "D_bus_idpop": BUS_FEATURES + ID_POPULATION_FEATURES,
    "E_all": BUS_FEATURES + ID_POPULATION_FEATURES + CROSS_ID_FEATURES,
}

ALL_GLOBAL_FEATURES: list[str] = (
    BUS_FEATURES + ID_POPULATION_FEATURES + CROSS_ID_FEATURES
)

# Streaming/causality classification for documentation.
FEATURE_CAUSALITY: dict[str, str] = {}
for _f in BUS_FEATURES + CROSS_ID_FEATURES:
    FEATURE_CAUSALITY[_f] = "causal"
for _f in ID_POPULATION_FEATURES:
    FEATURE_CAUSALITY[_f] = "causal"
# Features that additionally need a frozen calibration reference set.
for _f in ("g_new_id_count", "g_active_id_ratio", "g_freq_l1_to_calib",
           "g_id_dist_cosine_to_calib"):
    FEATURE_CAUSALITY[_f] = "causal_requires_calibration"


@dataclass
class GlobalFeatureConfig:
    """Configuration for the global feature pipeline."""

    window_frames: int = 200
    id_activity_frac: float = 0.05

    def as_dict(self) -> dict:
        return {
            "window_frames": self.window_frames,
            "id_activity_frac": self.id_activity_frac,
        }


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def _onehot(ids: np.ndarray, columns: list[str]) -> pd.DataFrame:
    onehot = pd.get_dummies(pd.Series(ids, dtype="string"))
    return onehot.reindex(columns=columns, fill_value=False).astype(float)


def build_global_features(
    window_table: pd.DataFrame,
    *,
    window_frames: int = 200,
    id_activity_frac: float = 0.05,
    calibration_ids: Iterable[str] | None = None,
    calib_id_freq: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compute global/cross-ID features aligned to ``window_table`` rows.

    Parameters
    ----------
    window_table
        Per-ID window feature table (chronological). Must contain ``timestamp``,
        ``can_id``, ``dlc``, ``byte_mean`` and ``iat_mean``.
    window_frames
        Trailing window length in emitted rows (one row == one CAN message
        after warm-up).
    id_activity_frac
        Frequency threshold used for ``g_n_ids_above_5pct``.
    calibration_ids
        IDs observed during calibration (for new-ID features). Frozen.
    calib_id_freq
        Calibration ID frequency distribution (for distribution-distance
        features). Frozen.

    Returns
    -------
    pd.DataFrame
        One row per input row, columns = :data:`ALL_GLOBAL_FEATURES` plus
        ``can_id``/``timestamp``/``is_attack``.
    """
    df = window_table.reset_index(drop=True)
    n = len(df)
    w = max(2, int(window_frames))
    if n == 0:
        return pd.DataFrame(columns=ALL_GLOBAL_FEATURES)

    ts = pd.Series(df["timestamp"].to_numpy(dtype=float))
    dlc = pd.Series(df["dlc"].to_numpy(dtype=float))
    byte_mean = pd.Series(df["byte_mean"].to_numpy(dtype=float))
    iat_mean = pd.Series(df["iat_mean"].to_numpy(dtype=float))
    ids = df["can_id"].astype(str).to_numpy()

    # ---- bus-level -------------------------------------------------------
    n_frames = ts.rolling(window=w, min_periods=1).count().to_numpy(dtype=float)
    duration = (ts - ts.shift(w - 1)).to_numpy(dtype=float, copy=True)
    leading = np.arange(n) < (w - 1)
    duration[leading] = ts.to_numpy()[leading] - ts.to_numpy()[0]
    duration = np.where(duration > 0, duration, np.nan)

    giat = ts.diff()
    riat = giat.rolling(window=w, min_periods=2)
    iat_mean_v = riat.mean()
    iat_std_v = riat.std(ddof=0)
    iat_min_v = riat.min()
    iat_max_v = riat.max()
    iat_median_v = riat.median()
    iat_cv = iat_std_v / (iat_mean_v + EPS)
    burstiness = (iat_std_v - iat_mean_v) / (iat_std_v + iat_mean_v + EPS)

    rdlc = dlc.rolling(window=w, min_periods=1)
    dlc_mean = rdlc.mean()
    dlc_std = rdlc.std(ddof=0)
    bus_bytes = dlc.rolling(window=w, min_periods=1).sum() / duration
    payload_bus_mean = byte_mean.rolling(window=w, min_periods=1).mean()
    frame_rate = (n_frames - 1.0) / duration

    # ---- ID population ---------------------------------------------------
    columns = sorted(set(ids.tolist()))
    onehot = _onehot(ids, columns)
    id_counts = onehot.rolling(window=w, min_periods=1).sum()
    counts_arr = id_counts.to_numpy(dtype=float)
    total = counts_arr.sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    p = counts_arr / safe_total[:, None]
    n_unique = (counts_arr > 0).sum(axis=1).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    entropy = -plogp.sum(axis=1)
    entropy_norm = np.where(
        n_unique > 1, entropy / np.log(np.maximum(n_unique, 2.0)), 0.0
    )
    concentration = np.where(n_unique > 1, 1.0 - entropy_norm, 0.0)
    dominant = counts_arr.max(axis=1) / safe_total
    sorted_counts = np.sort(counts_arr, axis=1)
    top3 = sorted_counts[:, -min(3, counts_arr.shape[1]):].sum(axis=1) / safe_total
    n_above = (p >= id_activity_frac).sum(axis=1).astype(float)

    cal_ids = set(str(c) for c in calibration_ids) if calibration_ids is not None else set()
    if cal_ids:
        cal_cols = [j for j, c in enumerate(columns) if c in cal_ids]
        new_mask = np.ones(len(columns), dtype=bool)
        new_mask[cal_cols] = False
        new_id_count = (
            (counts_arr[:, new_mask] > 0).sum(axis=1).astype(float)
            if new_mask.any()
            else np.zeros(n)
        )
        active_id_ratio = n_unique / max(1, len(cal_ids))
    else:
        new_id_count = np.zeros(n)
        active_id_ratio = np.zeros(n)

    # ---- cross-ID --------------------------------------------------------
    active = counts_arr > 0
    mean_active = safe_total / np.maximum(n_unique, 1.0)
    diff = np.where(active, counts_arr - mean_active[:, None], 0.0)
    var_active = (diff ** 2).sum(axis=1) / np.maximum(n_unique, 1.0)
    id_count_cv = np.sqrt(np.maximum(var_active, 0.0)) / (mean_active + EPS)

    if calib_id_freq:
        pc = np.array([float(calib_id_freq.get(c, 0.0)) for c in columns])
        pc_sum = pc.sum()
        pc = pc / pc_sum if pc_sum > 0 else pc
        freq_l1 = 0.5 * np.abs(p - pc[None, :]).sum(axis=1)
        pnorm = np.linalg.norm(p, axis=1)
        pc_norm = np.linalg.norm(pc)
        cos = (p @ pc) / (pnorm * pc_norm + EPS)
    else:
        freq_l1 = np.zeros(n)
        cos = np.zeros(n)

    # Per-ID mean byte/iat within the trailing window -> dispersion across IDs.
    byte_sum = onehot.mul(byte_mean, axis=0).rolling(window=w, min_periods=1).sum()
    iat_sum = onehot.mul(iat_mean, axis=0).rolling(window=w, min_periods=1).sum()
    byte_sum = byte_sum.to_numpy(dtype=float)
    iat_sum = iat_sum.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_id_byte = np.where(active, byte_sum / np.maximum(counts_arr, 1.0), np.nan)
        per_id_iat = np.where(active, iat_sum / np.maximum(counts_arr, 1.0), np.nan)
    payload_std_across_ids = np.nanstd(per_id_byte, axis=1)
    iat_std_across_ids = np.nanstd(per_id_iat, axis=1)

    out = pd.DataFrame(
        {
            "g_n_frames": n_frames,
            "g_duration_s": duration,
            "g_frame_rate": frame_rate,
            "g_iat_mean": iat_mean_v.to_numpy(dtype=float),
            "g_iat_std": iat_std_v.to_numpy(dtype=float),
            "g_iat_cv": iat_cv.to_numpy(dtype=float),
            "g_iat_min": iat_min_v.to_numpy(dtype=float),
            "g_iat_max": iat_max_v.to_numpy(dtype=float),
            "g_iat_median": iat_median_v.to_numpy(dtype=float),
            "g_burstiness": burstiness.to_numpy(dtype=float),
            "g_dlc_mean": dlc_mean.to_numpy(dtype=float),
            "g_dlc_std": dlc_std.to_numpy(dtype=float),
            "g_bus_bytes_per_sec_proxy": bus_bytes.to_numpy(dtype=float),
            "g_payload_bus_mean": payload_bus_mean.to_numpy(dtype=float),
            "g_n_unique_ids": n_unique,
            "g_new_id_count": new_id_count,
            "g_id_entropy": entropy,
            "g_id_entropy_norm": entropy_norm,
            "g_id_concentration": concentration,
            "g_dominant_id_frac": dominant,
            "g_top3_id_frac": top3,
            "g_n_ids_above_5pct": n_above,
            "g_active_id_ratio": active_id_ratio,
            "g_id_count_cv": id_count_cv,
            "g_freq_l1_to_calib": freq_l1,
            "g_id_dist_cosine_to_calib": cos,
            "g_payload_std_across_ids": payload_std_across_ids,
            "g_iat_std_across_ids": iat_std_across_ids,
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    # Leading rows have no complete trailing window; make the representation
    # well-defined (the normalizer would otherwise map them to 0 anyway).
    out[ALL_GLOBAL_FEATURES] = out[ALL_GLOBAL_FEATURES].fillna(0.0)
    out["can_id"] = ids
    out["timestamp"] = df["timestamp"].to_numpy()
    out["is_attack"] = df["is_attack"].to_numpy()
    if "attack_frac" in df.columns:
        out["attack_frac"] = df["attack_frac"].to_numpy()
    else:
        out["attack_frac"] = 0.0
    # Preserve row order / alignment with the per-ID window table.
    out.index = df.index
    return out


# ---------------------------------------------------------------------------
# Global normalization (not per-ID)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GlobalStats:
    """Frozen global normalizer, fit on calibration normals only."""

    center: dict[str, float]
    scale: dict[str, float]
    feature_cols: tuple[str, ...]
    method: str = "robust"
    provenance: str = "calib_normal"
    n_calib_normal_windows: int = 0

    def assert_provenance(self, expected: str = "calib_normal") -> None:
        if self.provenance != expected:
            raise AssertionError(
                f"global stats provenance {self.provenance!r} != {expected!r}; "
                "refusing to normalize with non-calibration statistics."
            )


def fit_global_stats(
    df_calib: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    method: str = "robust",
    provenance: str = "calib_normal",
) -> GlobalStats:
    """Fit global normalizer on calibration **normals only**."""
    if provenance != "calib_normal":
        raise ValueError(
            "GlobalStats may only be built from calibration normals "
            f"(got provenance={provenance!r})."
        )
    norm = df_calib[df_calib["is_attack"] == 0]
    center: dict[str, float] = {}
    scale: dict[str, float] = {}
    for f in feature_cols:
        x = pd.to_numeric(norm[f], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            center[f], scale[f] = 0.0, 1.0
            continue
        if method == "zscore":
            c = float(np.mean(x))
            s = float(np.std(x))
        elif method == "robust":
            c = float(np.median(x))
            iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
            s = iqr / 1.349 if iqr > 0 else float(np.std(x))
        else:
            raise ValueError(f"Unknown normalization method: {method!r}")
        if not np.isfinite(s) or s <= EPS:
            s = 1.0
        center[f], scale[f] = c, s
    return GlobalStats(
        center=center,
        scale=scale,
        feature_cols=tuple(feature_cols),
        method=method,
        provenance=provenance,
        n_calib_normal_windows=len(norm),
    )


def transform_global(
    df: pd.DataFrame,
    stats: GlobalStats,
    *,
    suffix: str = "_gres",
) -> pd.DataFrame:
    """Residualize global features: ``r = (x - center) / (scale + EPS)``."""
    stats.assert_provenance()
    feature_cols = list(stats.feature_cols)
    n = len(df)
    arr = df[feature_cols].to_numpy(dtype=float, copy=True)
    center = np.array([stats.center[f] for f in feature_cols], dtype=float)
    scale = np.array([stats.scale[f] for f in feature_cols], dtype=float)
    with np.errstate(invalid="ignore"):
        res = (arr - center) / (scale + EPS)
    res = np.where(np.isfinite(res), res, 0.0)
    out = {f + suffix: res[:, j] for j, f in enumerate(feature_cols)}
    out["can_id"] = df["can_id"].astype(str).values
    out["timestamp"] = df["timestamp"].values
    out["is_attack"] = df["is_attack"].values
    out["attack_frac"] = (
        df["attack_frac"].values if "attack_frac" in df.columns else np.zeros(n)
    )
    return pd.DataFrame(out)


__all__ = [
    "ALL_GLOBAL_FEATURES",
    "BUS_FEATURES",
    "CROSS_ID_FEATURES",
    "FEATURE_CAUSALITY",
    "GLOBAL_ABLATIONS",
    "GLOBAL_FEATURE_FAMILIES",
    "ID_POPULATION_FEATURES",
    "GlobalFeatureConfig",
    "GlobalStats",
    "build_global_features",
    "fit_global_stats",
    "transform_global",
]
