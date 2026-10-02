"""Unit tests for gradual-drift attack scheduling and application."""

from __future__ import annotations

import numpy as np
import pandas as pd

from canguard.attacks.adaptive import (
    DRIFT_RATE_FRACTION,
    AttackParameters,
    apply_alpha_series,
    gradual_alphas,
)

FEATURES = ["iat_mean", "byte_mean", "byte_var", "byte_nunique", "dlc_mode", "window_fill"]


def _params() -> AttackParameters:
    mu = {"iat_mean": 0.001, "byte_mean": 100.0, "byte_var": 400.0,
          "byte_nunique": 50.0, "dlc_mode": 8.0, "window_fill": 1.0}
    sd = {"iat_mean": 0.0002, "byte_mean": 10.0, "byte_var": 50.0,
          "byte_nunique": 5.0, "dlc_mode": 0.0, "window_fill": 0.0}
    return AttackParameters.from_stats({"0316": (mu, sd)}, (mu, sd), FEATURES)


def test_schedules_are_monotonic_and_bounded():
    for shape in ("linear", "exponential", "sqrt"):
        for rate in DRIFT_RATE_FRACTION:
            a = gradual_alphas(200, alpha_start=0.0, alpha_end=3.0, shape=shape, rate=rate)
            assert len(a) == 200
            assert np.all(np.diff(a) >= -1e-12)
            assert a[0] == 0.0
            assert a[-1] == 3.0


def test_attack_begins_at_zero_at_intended_time():
    a = gradual_alphas(100, alpha_start=0.0, alpha_end=2.0, shape="linear", rate="slow")
    assert a[0] == 0.0
    assert a[1] > a[0]


def test_drift_duration_matches_rate():
    n = 100
    for rate, frac in DRIFT_RATE_FRACTION.items():
        a = gradual_alphas(n, alpha_start=0.0, alpha_end=1.0, shape="linear", rate=rate)
        drift_len = max(1, int(round(frac * n)))
        # reaches the end severity at the drift length, then stays flat
        assert a[drift_len - 1] == 1.0
        assert np.allclose(a[drift_len - 1:], 1.0)


def test_explicit_fraction_schedule():
    fracs = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 1.0]
    a = gradual_alphas(140, alpha_start=0.0, alpha_end=2.0, fractions=fracs)
    assert len(a) == 140
    assert np.all(np.diff(a) >= -1e-12)
    assert a[-1] == 2.0
    assert a[0] == 0.02  # first fraction 0.01 * alpha_end 2.0


def test_gradual_application_only_touches_target_and_grows():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        cid = "0316" if i % 2 == 0 else "018f"
        rows.append({"can_id": cid, "timestamp": i * 0.001, "is_attack": 0, "attack_frac": 0.0,
                     "iat_mean": 0.001, "byte_mean": 100 + rng.normal(0, 3),
                     "byte_var": 400.0, "byte_nunique": 50.0, "dlc_mode": 8.0, "window_fill": 1.0})
    df = pd.DataFrame(rows)
    p = _params()
    target_idx = df.index[df["can_id"] == "0316"].to_numpy()
    alphas = gradual_alphas(len(target_idx), alpha_start=0.0, alpha_end=2.0,
                            shape="linear", rate="slow")
    out = apply_alpha_series(p, df, "0316", alphas, "positive", FEATURES, mode="additive")
    other = df["can_id"] != "0316"
    pd.testing.assert_frame_equal(out.loc[other].reset_index(drop=True),
                                  df.loc[other].reset_index(drop=True))
    # first attacked row unchanged (alpha 0), last attacked row larger than first
    first = out.loc[target_idx[0], "byte_mean"]
    last = out.loc[target_idx[-1], "byte_mean"]
    assert np.isclose(first, 100.0, atol=5.0)
    assert last > first


def test_alphas_length_must_match():
    df = pd.DataFrame({"can_id": ["0316"], "timestamp": [0.0], "is_attack": [0],
                       "iat_mean": [0.001], "byte_mean": [100.0], "byte_var": [1.0],
                       "byte_nunique": [1.0], "dlc_mode": [8.0], "window_fill": [1.0]})
    p = _params()
    with np.testing.assert_raises(ValueError):
        apply_alpha_series(p, df, "0316", [0.0, 1.0], "positive", FEATURES, mode="additive")
