"""Distribution-shift diagnostics for the global representation.

These utilities quantify how much each global feature shifts between
**calibration-normal** traffic and **test-normal** traffic, and whether the
shift is associated with false alarms. Only normal traffic is used for the
calibration-vs-test shift; attack samples are never involved in the shift
measurement.

Statistics: Kolmogorov-Smirnov ``D_KS``, Wasserstein ``W_1``, standardized mean
difference ``SMD`` (effect size), plus location/scale summaries and quantiles.
Ranking criterion: absolute ``SMD`` (primary), with scale-normalized
Wasserstein as a secondary magnitude measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-12

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _clean(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def standardized_mean_difference(cal, test) -> float:
    """Cohen's-d style effect size using the pooled standard deviation."""
    cal = _clean(cal)
    test = _clean(test)
    if len(cal) < 2 or len(test) < 2:
        return float("nan")
    pooled = np.sqrt((np.var(cal, ddof=1) + np.var(test, ddof=1)) / 2.0)
    if pooled <= EPS:
        diff = float(np.mean(test) - np.mean(cal))
        if abs(diff) <= EPS:
            return 0.0
        return float(np.sign(diff)) * float("inf")
    return float((np.mean(test) - np.mean(cal)) / pooled)


def distribution_shift(cal, test) -> dict:
    """Location/scale, effect size and distance statistics for two samples."""
    cal = _clean(cal)
    test = _clean(test)
    out = {
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
        "cal_mean": float(np.mean(cal)) if len(cal) else float("nan"),
        "test_mean": float(np.mean(test)) if len(test) else float("nan"),
        "cal_std": float(np.std(cal, ddof=1)) if len(cal) > 1 else float("nan"),
        "test_std": float(np.std(test, ddof=1)) if len(test) > 1 else float("nan"),
        "cal_median": float(np.median(cal)) if len(cal) else float("nan"),
        "test_median": float(np.median(test)) if len(test) else float("nan"),
    }
    for q in QUANTILES:
        out[f"cal_q{int(q * 100)}"] = float(np.quantile(cal, q)) if len(cal) else float("nan")
        out[f"test_q{int(q * 100)}"] = float(np.quantile(test, q)) if len(test) else float("nan")
    out["smd"] = standardized_mean_difference(cal, test)
    if len(cal) < 2 or len(test) < 2:
        out["ks"] = float("nan")
        out["wasserstein"] = float("nan")
        out["normalized_wasserstein"] = float("nan")
        return out
    out["ks"] = float(stats.ks_2samp(cal, test).statistic)
    w1 = float(stats.wasserstein_distance(cal, test))
    pooled = np.sqrt((np.var(cal, ddof=1) + np.var(test, ddof=1)) / 2.0)
    out["wasserstein"] = w1
    out["normalized_wasserstein"] = float(w1 / (pooled + EPS))
    return out


def feature_shift_table(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    family_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Per-feature normal calibration-vs-test shift table, ranked by |SMD|."""
    family_of = {f: fam for fam, feats in family_map.items() for f in feats}
    rows = []
    for f in feature_cols:
        row = {"feature": f, "family": family_of.get(f, "unknown")}
        row.update(distribution_shift(cal_df[f], test_df[f]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["abs_smd"] = df["smd"].abs()
    return df.sort_values(
        ["abs_smd", "normalized_wasserstein"], ascending=False
    ).reset_index(drop=True)


def family_shift_summary(feature_shift: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shift magnitude by feature family."""
    if feature_shift.empty:
        return pd.DataFrame()
    g = feature_shift.groupby("family")
    out = g.agg(
        n_features=("feature", "count"),
        mean_abs_smd=("abs_smd", "mean"),
        median_abs_smd=("abs_smd", "median"),
        max_abs_smd=("abs_smd", "max"),
        mean_ks=("ks", "mean"),
        mean_wasserstein=("wasserstein", "mean"),
        mean_normalized_wasserstein=("normalized_wasserstein", "mean"),
    ).reset_index()
    return out.sort_values("mean_abs_smd", ascending=False).reset_index(drop=True)


def association_with_flags(
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_flags: np.ndarray,
    test_scores: np.ndarray,
    feature_cols: list[str],
    family_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Associate each feature's (normal) deviation with false alarms.

    Only test-normal windows are used. For each feature we report the effect
    size between flagged and non-flagged normal windows, the Spearman rank
    correlation between the feature's calibration-standardized deviation and
    the anomaly score, and the rate at which flagged windows exceed the
    calibration mean +/- 2 sd.
    """
    family_of = {f: fam for fam, feats in family_map.items() for f in feats}
    flags = np.asarray(test_flags).astype(bool)
    scores = np.asarray(test_scores, dtype=float)
    rows = []
    for f in feature_cols:
        cal = _clean(cal_df[f])
        tv = np.asarray(test_df[f], dtype=float)
        mu = float(np.mean(cal)) if len(cal) else 0.0
        sd = float(np.std(cal, ddof=1)) if len(cal) > 1 else 1.0
        if not np.isfinite(sd) or sd <= EPS:
            sd = 1.0
        z = (tv - mu) / sd
        finite = np.isfinite(z)
        flagged = flags & finite
        normal = (~flags) & finite
        if flagged.any() and normal.any():
            smd_flag = standardized_mean_difference(z[normal], z[flagged])
        else:
            smd_flag = float("nan")
        if finite.sum() > 2 and len(np.unique(scores[finite])) > 1:
            rho = float(stats.spearmanr(z[finite], scores[finite]).statistic)
        else:
            rho = float("nan")
        exceed = float(np.mean(np.abs(z[flagged]) > 2.0)) if flagged.any() else float("nan")
        rows.append({
            "feature": f,
            "family": family_of.get(f, "unknown"),
            "smd_flagged_vs_notflagged": smd_flag,
            "abs_smd_flagged": abs(smd_flag) if np.isfinite(smd_flag) else float("nan"),
            "spearman_deviation_vs_score": rho,
            "flagged_exceed_2sd_rate": exceed,
            "n_flagged": int(flagged.sum()),
            "n_normal": int(normal.sum()),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("abs_smd_flagged", ascending=False).reset_index(drop=True)


__all__ = [
    "QUANTILES",
    "association_with_flags",
    "distribution_shift",
    "family_shift_summary",
    "feature_shift_table",
    "standardized_mean_difference",
]
