"""Leakage tests for the frozen ROAD per-capture pre-injection protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd

from canguard.exp.road_protocol import build_road_raw_residual_splits
from canguard.features.groups import BEHAVIORAL_FEATURES_V1

FEATURES = list(BEHAVIORAL_FEATURES_V1)


def _road_window_table(seed: int = 0, n_pre: int = 120, n_test: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ts = 0.0
    for i in range(n_pre):
        r = {"timestamp": ts, "elapsed": ts, "can_id": ["0316", "018f"][i % 2],
             "is_attack": 0, "attack_frac": 0.0}
        r.update({f: float(rng.normal(5.0, 1.0)) for f in FEATURES})
        rows.append(r)
        ts += 0.01
    for i in range(n_test):
        cid = ["0316", "018f"][i % 2]
        attack = 1 if (cid == "0316" and i > 20) else 0
        r = {"timestamp": ts, "elapsed": ts, "can_id": cid,
             "is_attack": attack, "attack_frac": float(attack)}
        r.update({f: float(rng.normal(5.0, 1.0)) for f in FEATURES})
        rows.append(r)
        ts += 0.01
    return pd.DataFrame(rows)


def test_split_is_pre_injection_vs_post():
    ft = _road_window_table()
    start = 1.2
    out = build_road_raw_residual_splits(ft, [start, start + 0.5], FEATURES)
    assert out["raw"]["train"]["elapsed"].max() < start
    assert out["raw"]["test"]["elapsed"].min() >= start
    assert int(out["raw"]["train"]["is_attack"].sum()) == 0
    assert out["n_test_attack"] > 0


def test_test_attack_values_do_not_change_calibration_residuals():
    ft = _road_window_table()
    out1 = build_road_raw_residual_splits(ft, [1.2, 1.7], FEATURES)
    perturbed = ft.copy()
    test_attack = (perturbed["elapsed"] >= 1.2) & (perturbed["is_attack"] == 1)
    perturbed.loc[test_attack, FEATURES] = 1e6
    out2 = build_road_raw_residual_splits(perturbed, [1.2, 1.7], FEATURES)
    # Calibration residuals and fitted stats are unchanged by test attacks.
    pd.testing.assert_frame_equal(
        out1["residual"]["train"].reset_index(drop=True),
        out2["residual"]["train"].reset_index(drop=True),
    )
    assert out1["per_id_stats_n"] == out2["per_id_stats_n"]


def test_test_attack_can_change_test_residuals_but_not_train():
    ft = _road_window_table()
    out1 = build_road_raw_residual_splits(ft, [1.2, 1.7], FEATURES)
    perturbed = ft.copy()
    test_attack = (perturbed["elapsed"] >= 1.2) & (perturbed["is_attack"] == 1)
    perturbed.loc[test_attack, FEATURES] = -1e6
    out2 = build_road_raw_residual_splits(perturbed, [1.2, 1.7], FEATURES)
    # train identical; test residual differs where attacks were perturbed.
    pd.testing.assert_frame_equal(
        out1["residual"]["train"].reset_index(drop=True),
        out2["residual"]["train"].reset_index(drop=True),
    )
    assert not np.allclose(
        out1["residual"]["test"][FEATURES[0] + "_res"].to_numpy(),
        out2["residual"]["test"][FEATURES[0] + "_res"].to_numpy(),
    )


def test_too_few_pre_windows_returns_error():
    ft = _road_window_table(n_pre=10, n_test=100)
    out = build_road_raw_residual_splits(ft, [0.05, 0.5], FEATURES, min_pre_windows=50)
    assert "error" in out
