"""Tests for global normalization and provenance/leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canguard.features.global_features import (
    GlobalStats,
    fit_global_stats,
    transform_global,
)

FEATURES = ["g_frame_rate", "g_id_entropy", "g_payload_bus_mean"]


def _df(n: int = 80, seed: int = 0, attack_from: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        attack = 1 if (attack_from is not None and i >= attack_from) else 0
        rows.append({
            "can_id": "a", "timestamp": float(i), "is_attack": attack, "attack_frac": float(attack),
            "g_frame_rate": 100.0 + rng.normal(0, 5),
            "g_id_entropy": 2.0 + rng.normal(0, 0.1),
            "g_payload_bus_mean": 50.0 + rng.normal(0, 2),
        })
    return pd.DataFrame(rows)


def test_fit_ignores_attack_rows():
    df = _df(attack_from=40)
    s1 = fit_global_stats(df, FEATURES, method="robust")
    perturbed = df.copy()
    atk = perturbed["is_attack"] == 1
    perturbed.loc[atk, FEATURES] = 1e6
    s2 = fit_global_stats(perturbed, FEATURES, method="robust")
    for f in FEATURES:
        assert s1.center[f] == pytest.approx(s2.center[f])
        assert s1.scale[f] == pytest.approx(s2.scale[f])


def test_provenance_guard():
    df = _df()
    with pytest.raises(ValueError):
        fit_global_stats(df, FEATURES, provenance="test")
    s = fit_global_stats(df, FEATURES)
    with pytest.raises(AssertionError):
        transform_global(df, GlobalStats(center=s.center, scale=s.scale,
                                         feature_cols=s.feature_cols, provenance="test"))


def test_reproducible_and_transform_centered():
    df = _df()
    s1 = fit_global_stats(df, FEATURES, method="zscore")
    s2 = fit_global_stats(df, FEATURES, method="zscore")
    assert s1.center == s2.center
    out = transform_global(df, s1)
    # z-score normalizer should make the normal calibration near zero-mean.
    normal = out[out["is_attack"] == 0]
    for f in FEATURES:
        assert abs(float(normal[f + "_gres"].mean())) < 0.5


def test_constant_feature_handled():
    df = _df()
    df["g_frame_rate"] = 7.0
    s = fit_global_stats(df, FEATURES, method="robust")
    assert s.scale["g_frame_rate"] == 1.0
    out = transform_global(df, s)
    assert np.isfinite(out["g_frame_rate_gres"]).all()


def test_missing_feature_values_handled():
    df = _df()
    df.loc[0, "g_id_entropy"] = np.nan
    s = fit_global_stats(df, FEATURES, method="robust")
    out = transform_global(df, s)
    assert np.isfinite(out["g_id_entropy_gres"]).all()


def test_robust_vs_zscore_scale():
    df = _df()
    z = fit_global_stats(df, FEATURES, method="zscore")
    r = fit_global_stats(df, FEATURES, method="robust")
    # both finite and positive
    for f in FEATURES:
        assert z.scale[f] > 0
        assert r.scale[f] > 0
