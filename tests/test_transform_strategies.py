"""Tests for pluggable residualization strategies (Phase 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canguard.transforms import (
    RESIDUALIZERS,
    build_residualizer,
    list_residualizers,
)

FEATURES = ["iat_mean", "byte_mean", "window_fill", "time_since_last_seen"]


def _frame(rng: np.random.Generator, n_per_id: int = 120) -> pd.DataFrame:
    ids = ["0316", "018f", "02b0"]
    rows = []
    ts = 0.0
    for cid in ids:
        base = rng.normal(0, 1, size=4)
        for _ in range(n_per_id):
            ts += 0.001
            rows.append({
                "can_id": cid,
                "timestamp": ts,
                "is_attack": 0,
                "attack_frac": 0.0,
                "iat_mean": base[0] + rng.normal(0, 0.1),
                "byte_mean": base[1] + rng.normal(0, 0.2),
                "window_fill": 1.0,
                "time_since_last_seen": base[3] + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def test_factory_lists_and_rejects_unknown():
    assert set(list_residualizers()) == set(RESIDUALIZERS)
    assert set(list_residualizers()) == {
        "raw", "global_z", "per_id_z", "per_id_mad", "per_id_quantile",
    }
    with pytest.raises(ValueError):
        build_residualizer("nope")


@pytest.mark.parametrize("name", list(RESIDUALIZERS))
def test_transform_produces_residual_schema(name):
    rng = np.random.default_rng(0)
    calib = _frame(rng)
    test = _frame(rng)
    res = build_residualizer(name)
    res.fit(calib, FEATURES)
    out = res.transform(test, FEATURES)
    for f in FEATURES:
        assert f + "_res" in out.columns
    for col in ("can_id", "timestamp", "is_attack"):
        assert col in out.columns
    matrix = out[[f + "_res" for f in FEATURES]].to_numpy()
    assert np.isfinite(matrix).all()


def test_raw_preserves_feature_values():
    rng = np.random.default_rng(1)
    calib = _frame(rng)
    test = _frame(rng)
    out = build_residualizer("raw").fit_transform(calib, test, FEATURES)
    np.testing.assert_allclose(out["iat_mean_res"].to_numpy(),
                               test["iat_mean"].to_numpy())


def test_per_id_z_centers_each_id():
    rng = np.random.default_rng(2)
    calib = _frame(rng, n_per_id=200)
    res = build_residualizer("per_id_z")
    res.fit(calib, FEATURES)
    out = res.transform(calib, FEATURES)
    for cid, g in out.groupby("can_id"):
        mu = g["iat_mean_res"].mean()
        sd = g["iat_mean_res"].std()
        assert abs(mu) < 1e-9, cid
        assert 0.5 < sd < 2.0, cid


def test_mad_residual_median_is_zero_per_id():
    rng = np.random.default_rng(3)
    calib = _frame(rng, n_per_id=200)
    res = build_residualizer("per_id_mad")
    res.fit(calib, FEATURES)
    out = res.transform(calib, FEATURES)
    for _cid, g in out.groupby("can_id"):
        assert abs(float(np.median(g["byte_mean_res"]))) < 1e-9


def test_quantile_maps_median_near_zero():
    rng = np.random.default_rng(4)
    calib = _frame(rng, n_per_id=300)
    res = build_residualizer("per_id_quantile")
    res.fit(calib, FEATURES)
    out = res.transform(calib, FEATURES)
    for _cid, g in out.groupby("can_id"):
        assert abs(float(np.median(g["byte_mean_res"]))) < 0.5


def test_per_id_strategies_fall_back_for_unseen_id():
    rng = np.random.default_rng(5)
    calib = _frame(rng)
    unseen = _frame(rng).copy()
    unseen["can_id"] = "0x999"
    for name in ("per_id_z", "per_id_mad", "per_id_quantile"):
        res = build_residualizer(name)
        res.fit(calib, FEATURES)
        out = res.transform(unseen, FEATURES)
        assert np.isfinite(out[[f + "_res" for f in FEATURES]].to_numpy()).all()


def test_handles_missing_values():
    rng = np.random.default_rng(6)
    calib = _frame(rng)
    test = _frame(rng)
    test.loc[test.index[:5], "iat_mean"] = np.nan
    for name in list_residualizers():
        out = build_residualizer(name).fit_transform(calib, test, FEATURES)
        assert np.isfinite(out["iat_mean_res"].to_numpy()).all()
