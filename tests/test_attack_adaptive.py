"""Unit tests for the white-box adaptive attacker (fixed severity)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canguard.attacks.adaptive import (
    FEATURE_BOUNDS,
    INTEGER_FEATURES,
    AttackParameters,
    apply_fixed_attack,
    attack_feature_matrix,
    resolve_target_features,
)

FEATURES = ["iat_mean", "byte_mean", "byte_var", "byte_nunique", "dlc_mode", "window_fill"]


def _params() -> AttackParameters:
    mu = {"iat_mean": 0.001, "byte_mean": 100.0, "byte_var": 400.0,
          "byte_nunique": 50.0, "dlc_mode": 8.0, "window_fill": 1.0}
    sd = {"iat_mean": 0.0002, "byte_mean": 10.0, "byte_var": 50.0,
          "byte_nunique": 5.0, "dlc_mode": 0.0, "window_fill": 0.0}
    per_id = {"0316": (mu, sd)}
    return AttackParameters.from_stats(per_id, (mu, sd), FEATURES)


def _frame(seed: int = 0, n_target: int = 60, n_other: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ts = 0.0
    for i in range(n_target):
        rows.append({"can_id": "0316", "timestamp": ts, "is_attack": 0, "attack_frac": 0.0,
                     "iat_mean": 0.001 + rng.normal(0, 1e-4), "byte_mean": 100 + rng.normal(0, 5),
                     "byte_var": 400 + rng.normal(0, 10), "byte_nunique": 50.0,
                     "dlc_mode": 8.0, "window_fill": 1.0})
        ts += 0.001
    for i in range(n_other):
        rows.append({"can_id": "018f", "timestamp": ts, "is_attack": 0, "attack_frac": 0.0,
                     "iat_mean": 0.002, "byte_mean": 12.0, "byte_var": 3.0,
                     "byte_nunique": 4.0, "dlc_mode": 8.0, "window_fill": 1.0})
        ts += 0.001
    return pd.DataFrame(rows)


def test_severity_zero_is_no_modification_both_modes():
    df = _frame()
    p = _params()
    for mode in ("mean_shift", "additive"):
        out = apply_fixed_attack(p, df, "0316", 0.0, "positive", FEATURES, mode=mode)
        pd.testing.assert_frame_equal(out, df)


def test_positive_and_negative_directions_mean_shift():
    df = _frame()
    p = _params()
    pos = apply_fixed_attack(p, df, "0316", 1.0, "positive", FEATURES, mode="mean_shift")
    neg = apply_fixed_attack(p, df, "0316", 1.0, "negative", FEATURES, mode="mean_shift")
    target = df["can_id"] == "0316"
    # mean_shift ignores current values: positive >= negative on every bound-free feature.
    assert (pos.loc[target, "byte_mean"] >= neg.loc[target, "byte_mean"]).all()
    assert (pos.loc[target, "byte_mean"] >= p.global_mu["byte_mean"]).all()
    assert (neg.loc[target, "byte_mean"] <= p.global_mu["byte_mean"]).all()


def test_additive_direction_moves_relative_to_current():
    df = _frame(seed=3)
    p = _params()
    pos = apply_fixed_attack(p, df, "0316", 2.0, "positive", FEATURES, mode="additive")
    neg = apply_fixed_attack(p, df, "0316", 2.0, "negative", FEATURES, mode="additive")
    target = (df["can_id"] == "0316").to_numpy()
    assert (pos.loc[target, "byte_mean"].to_numpy() > df.loc[target, "byte_mean"].to_numpy()).all()
    assert (neg.loc[target, "byte_mean"].to_numpy() < df.loc[target, "byte_mean"].to_numpy()).all()


def test_bounds_and_integer_constraints_respected():
    df = _frame()
    p = _params()
    for direction in ("positive", "negative"):
        out = apply_fixed_attack(p, df, "0316", 100.0, direction, FEATURES, mode="mean_shift")
        target = out["can_id"] == "0316"
        for f in FEATURES:
            lo, hi = FEATURE_BOUNDS[f]
            col = out.loc[target, f]
            if lo is not None:
                assert (col >= lo - 1e-9).all()
            if hi is not None:
                assert (col <= hi + 1e-9).all()
            if f in INTEGER_FEATURES:
                assert np.allclose(col, np.round(col))


def test_only_target_id_is_modified():
    df = _frame()
    p = _params()
    out = apply_fixed_attack(p, df, "0316", 1.5, "positive", FEATURES, mode="mean_shift")
    other = df["can_id"] != "0316"
    pd.testing.assert_frame_equal(out.loc[other].reset_index(drop=True),
                                  df.loc[other].reset_index(drop=True))


def test_attack_matrix_zero_variance_feature_stays_at_mean():
    p = _params()
    X = attack_feature_matrix(p, "0316", np.array([3.0]), "positive", ["dlc_mode"],
                              mode="mean_shift")
    assert X[0, 0] == p.global_mu["dlc_mode"]


def test_resolve_target_features():
    assert "window_fill" not in resolve_target_features(None)
    assert set(resolve_target_features("iat")) == {
        "iat_mean", "iat_std", "iat_median", "iat_min", "iat_max"
    }
    explicit = resolve_target_features(["byte_mean", "not_a_feature"], FEATURES)
    assert explicit == ["byte_mean"]


def test_provenance_guard_rejects_test_statistics():
    mu = {f: 0.0 for f in FEATURES}
    sd = {f: 1.0 for f in FEATURES}
    with pytest.raises(ValueError, match="calibration normals"):
        AttackParameters.from_stats({"0316": (mu, sd)}, (mu, sd), FEATURES,
                                    provenance="test")
    p = _params()
    with pytest.raises(AssertionError):
        p.assert_provenance("test")
