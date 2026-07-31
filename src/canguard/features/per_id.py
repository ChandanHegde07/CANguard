"""Per-ID residual transformation (PIRD core).

Fits per-ID mean/std from normal calibration windows, then normalizes any
window by its ID's baseline (z-score). IDs unseen in calibration fall back to
global normal statistics. NaNs in source features become 0.0 residuals.

Port of ``fit_per_id_stats`` / ``transform_residuals`` from ``pird_hcrl.ipynb``
-- no algorithm changes.
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
    """Fit per-ID and global normal statistics from calibration windows.

    Parameters
    ----------
    df_calib : pd.DataFrame
        Calibration window table containing ``is_attack`` and ``can_id`` columns.
    feature_cols : list[str]
        Feature columns to compute statistics over.

    Returns
    -------
    per_id_stats : PerIdStats
        ``{can_id: (mean_dict, std_dict)}`` for IDs with at least
        ``MIN_WINDOWS_PER_ID`` normal windows.
    global_stats : GlobalStats
        ``(mean_dict, std_dict)`` over all calibration normals (fallback).
    """
    norm = df_calib[df_calib["is_attack"] == 0].copy()
    stats: PerIdStats = {}
    for cid, grp in norm.groupby("can_id"):
        if len(grp) >= MIN_WINDOWS_PER_ID:
            mu = {f: grp[f].mean() for f in feature_cols}
            sd = {f: grp[f].std() for f in feature_cols}
            stats[cid] = (mu, sd)
    gmu = {f: norm[f].mean() for f in feature_cols}
    gsd = {f: norm[f].std() for f in feature_cols}
    return stats, (gmu, gsd)


def transform_residuals(
    df: pd.DataFrame,
    per_id_stats: PerIdStats,
    global_stats: GlobalStats,
    feature_cols: list[str],
    suffix: str = "_res",
) -> pd.DataFrame:
    """Residualize ``df`` features against per-ID (or global) baselines.

    For each row, if its ``can_id`` was fitted per-ID, use that mean/std;
    otherwise use the global fallback. Residual ``r = (x - mu) / (std + EPS)``.
    NaN source values are mapped to ``0.0``. Metadata columns (``can_id``,
    ``timestamp``, ``is_attack``, ``attack_frac``) are carried through.

    Parameters
    ----------
    df : pd.DataFrame
        Window table with rows to residualize.
    per_id_stats : PerIdStats
        Output of :func:`fit_per_id_stats`.
    global_stats : GlobalStats
        Output of :func:`fit_per_id_stats`.
    feature_cols : list[str]
        Feature columns to residualize.
    suffix : str
        Suffix appended to residual column names (default ``"_res"``).

    Returns
    -------
    pd.DataFrame
        Residualized table: one ``f + suffix`` column per ``feature_cols`` entry,
        plus the metadata columns.
    """
    gmu, gsd = global_stats
    rows = []
    for _, row in df.iterrows():
        cid = row["can_id"]
        if cid in per_id_stats:
            mu, sd = per_id_stats[cid]
        else:
            mu, sd = gmu, gsd
        r: dict = {}
        for f in feature_cols:
            val = row.get(f, np.nan)
            if pd.isna(val):
                r[f + suffix] = 0.0
            else:
                r[f + suffix] = (val - mu[f]) / (sd[f] + EPS)
        r["can_id"] = cid
        r["timestamp"] = row["timestamp"]
        r["is_attack"] = row["is_attack"]
        r["attack_frac"] = row.get("attack_frac", 0.0)
        rows.append(r)
    return pd.DataFrame(rows)
