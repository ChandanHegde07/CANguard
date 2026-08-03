"""Per-ID residual transformation (PIRD core).

Fits per-ID mean/std from normal calibration windows, then normalizes any
window by its ID's baseline (z-score). IDs unseen in calibration fall back to
global normal statistics. NaNs in source features become 0.0 residuals.

Vectorized implementation; numerically equivalent to the notebook port.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Minimum number of normal windows required to fit an ID-specific baseline.
MIN_WINDOWS_PER_ID = 20
# Additive constant to avoid zero-variance division.
EPS = 1e-6

# Mapping of CAN ID -> (feature->mean dict, feature->std dict)
PerIdStats = dict[str, tuple[dict[str, float], dict[str, float]]]
GlobalStats = tuple[dict[str, float], dict[str, float]]


def fit_per_id_stats(
    df_calib: pd.DataFrame, feature_cols: list[str]
) -> tuple[PerIdStats, GlobalStats]:
    """Fit per-ID and global normal statistics from calibration windows."""
    norm = df_calib[df_calib["is_attack"] == 0]
    stats: PerIdStats = {}
    for cid, grp in norm.groupby("can_id"):
        if len(grp) >= MIN_WINDOWS_PER_ID:
            mu = {f: float(grp[f].mean()) for f in feature_cols}
            sd = {f: float(grp[f].std()) for f in feature_cols}
            stats[str(cid)] = (mu, sd)
    gmu = {f: float(norm[f].mean()) for f in feature_cols}
    gsd = {f: float(norm[f].std()) for f in feature_cols}
    return stats, (gmu, gsd)


def transform_residuals(
    df: pd.DataFrame,
    per_id_stats: PerIdStats,
    global_stats: GlobalStats,
    feature_cols: list[str],
    suffix: str = "_res",
) -> pd.DataFrame:
    """Residualize ``df`` features against per-ID (or global) baselines.

    Residual ``r = (x - mu) / (std + EPS)``. NaN source values → 0.0.
    """
    gmu, gsd = global_stats
    n = len(df)
    # Precompute global arrays
    gmu_arr = np.array([gmu[f] for f in feature_cols], dtype=float)
    gsd_arr = np.array([gsd[f] for f in feature_cols], dtype=float)

    # Map each unique can_id to mean/std arrays
    ids = df["can_id"].astype(str).values
    unique_ids = pd.unique(ids)
    id_to_mu: dict[str, np.ndarray] = {}
    id_to_sd: dict[str, np.ndarray] = {}
    for cid in unique_ids:
        if cid in per_id_stats:
            mu_d, sd_d = per_id_stats[cid]
            id_to_mu[cid] = np.array([mu_d[f] for f in feature_cols], dtype=float)
            id_to_sd[cid] = np.array([sd_d[f] for f in feature_cols], dtype=float)
        else:
            id_to_mu[cid] = gmu_arr
            id_to_sd[cid] = gsd_arr

    X = df[feature_cols].to_numpy(dtype=float, copy=True)
    mu_mat = np.vstack([id_to_mu[c] for c in ids])
    sd_mat = np.vstack([id_to_sd[c] for c in ids])

    with np.errstate(invalid="ignore"):
        R = (X - mu_mat) / (sd_mat + EPS)
    R = np.where(np.isnan(R), 0.0, R)

    out = {f + suffix: R[:, j] for j, f in enumerate(feature_cols)}
    out["can_id"] = ids
    out["timestamp"] = df["timestamp"].values
    out["is_attack"] = df["is_attack"].values
    if "attack_frac" in df.columns:
        out["attack_frac"] = df["attack_frac"].values
    else:
        out["attack_frac"] = np.zeros(n, dtype=float)
    return pd.DataFrame(out)
