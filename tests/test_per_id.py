"""Equivalence tests for the residual transform vs notebook implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canguard.features.groups import BEHAVIORAL_FEATURES_V1
from canguard.features.per_id import EPS, MIN_WINDOWS_PER_ID, fit_per_id_stats, transform_residuals


# ---------------------------------------------------------------------------
# Reference (verbatim from pird_hcrl.ipynb)
# ---------------------------------------------------------------------------
def _ref_fit_per_id_stats(df_calib, feature_cols):
    norm = df_calib[df_calib["is_attack"] == 0].copy()
    stats = {}
    for cid, grp in norm.groupby("can_id"):
        if len(grp) >= MIN_WINDOWS_PER_ID:
            mu = {f: grp[f].mean() for f in feature_cols}
            sd = {f: grp[f].std() for f in feature_cols}
            stats[cid] = (mu, sd)
    gmu = {f: norm[f].mean() for f in feature_cols}
    gsd = {f: norm[f].std() for f in feature_cols}
    return stats, (gmu, gsd)


def _ref_transform_residuals(df, per_id_stats, global_stats, feature_cols):
    gmu, gsd = global_stats
    rows = []
    for _, row in df.iterrows():
        cid = row["can_id"]
        if cid in per_id_stats:
            mu, sd = per_id_stats[cid]
        else:
            mu, sd = gmu, gsd
        r = {}
        for f in feature_cols:
            val = row.get(f, np.nan)
            if pd.isna(val):
                r[f + "_res"] = 0.0
            else:
                r[f + "_res"] = (val - mu[f]) / (sd[f] + EPS)
        r["can_id"] = cid
        r["timestamp"] = row["timestamp"]
        r["is_attack"] = row["is_attack"]
        r["attack_frac"] = row.get("attack_frac", 0.0)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _synth_window_table(seed=7):
    rng = np.random.default_rng(seed)
    ids = ["0316", "018f", "02b0"]
    rows = []
    ts = 1478191030.0
    for _ in range(300):
        cid = ids[rng.integers(0, len(ids))]
        feat = {f: float(rng.normal(loc=5.0, scale=1.0)) for f in BEHAVIORAL_FEATURES_V1}
        attack = 1 if rng.random() < 0.15 else 0
        r = {
            "can_id": cid,
            "timestamp": ts,
            "is_attack": attack,
            "attack_frac": float(attack),
        }
        r.update(feat)
        rows.append(r)
        ts += float(rng.uniform(0.0002, 0.002))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_fit_per_id_stats_equivalent():
    df = _synth_window_table()
    got_stats, got_global = fit_per_id_stats(df, BEHAVIORAL_FEATURES_V1)
    ref_stats, ref_global = _ref_fit_per_id_stats(df, BEHAVIORAL_FEATURES_V1)
    assert set(got_stats.keys()) == set(ref_stats.keys())
    for cid in got_stats:
        for f in BEHAVIORAL_FEATURES_V1:
            assert got_stats[cid][0][f] == pytest.approx(ref_stats[cid][0][f])
            assert got_stats[cid][1][f] == pytest.approx(ref_stats[cid][1][f])
    for f in BEHAVIORAL_FEATURES_V1:
        assert got_global[0][f] == pytest.approx(ref_global[0][f])
        assert got_global[1][f] == pytest.approx(ref_global[1][f])


def test_transform_residuals_equivalent():
    df = _synth_window_table()
    stats, global_stats = fit_per_id_stats(df, BEHAVIORAL_FEATURES_V1)
    got = transform_residuals(df, stats, global_stats, BEHAVIORAL_FEATURES_V1)
    ref = _ref_transform_residuals(df, stats, global_stats, BEHAVIORAL_FEATURES_V1)
    pd.testing.assert_frame_equal(got, ref)


def test_unseen_id_uses_global_fallback():
    df = _synth_window_table()
    stats, global_stats = fit_per_id_stats(df, BEHAVIORAL_FEATURES_V1)
    novel = df.iloc[[0]].copy()
    novel.loc[:, "can_id"] = "0000"
    out = transform_residuals(novel, stats, global_stats, BEHAVIORAL_FEATURES_V1)
    mu, sd = global_stats
    expected = (novel.iloc[0]["iat_mean"] - mu["iat_mean"]) / (sd["iat_mean"] + EPS)
    assert out.iloc[0]["iat_mean_res"] == pytest.approx(expected)


def test_nan_feature_becomes_zero_residual():
    df = _synth_window_table().iloc[[0]].copy()
    df.loc[df.index[0], "iat_std"] = np.nan
    stats, global_stats = fit_per_id_stats(df, BEHAVIORAL_FEATURES_V1)
    out = transform_residuals(df, stats, global_stats, BEHAVIORAL_FEATURES_V1)
    assert out.iloc[0]["iat_std_res"] == 0.0


def test_per_id_stats_require_min_windows():
    df = _synth_window_table()
    small = df.iloc[:15]
    stats, _ = fit_per_id_stats(small, BEHAVIORAL_FEATURES_V1)
    assert stats == {}


def test_zero_variance_protected_by_eps():
    df = pd.DataFrame(
        {
            "can_id": ["0316"] * 40,
            "timestamp": np.linspace(0, 1, 40),
            "is_attack": [0] * 40,
            "attack_frac": [0.0] * 40,
            "iat_mean": [0.5] * 40,  # constant feature -> zero std
        }
    )
    stats, _ = fit_per_id_stats(df, ["iat_mean"])
    assert stats["0316"][1]["iat_mean"] == 0.0
    out = transform_residuals(df, stats, ({"iat_mean": 0.5}, {"iat_mean": 0.0}), ["iat_mean"])
    assert np.isfinite(out.iloc[0]["iat_mean_res"])
