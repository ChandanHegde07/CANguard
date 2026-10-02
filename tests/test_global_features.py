"""Tests for the causal global / cross-ID feature pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from canguard.features.global_features import (
    ALL_GLOBAL_FEATURES,
    BUS_FEATURES,
    CROSS_ID_FEATURES,
    ID_POPULATION_FEATURES,
    build_global_features,
)


def _table(n: int = 60, ids=("a", "b"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ts = 0.0
    for i in range(n):
        cid = ids[i % len(ids)]
        rows.append(
            {
                "can_id": cid,
                "timestamp": ts,
                "dlc": 8,
                "byte_mean": 100.0 + rng.normal(0, 1),
                "iat_mean": 0.001,
                "is_attack": 0,
                "attack_frac": 0.0,
            }
        )
        ts += 0.001
    return pd.DataFrame(rows)


def test_deterministic_extraction():
    df = _table()
    a = build_global_features(df, window_frames=10)
    b = build_global_features(df, window_frames=10)
    pd.testing.assert_frame_equal(a, b)


def test_frame_counts_and_unique_ids():
    df = _table(20, ids=("a", "b"))
    g = build_global_features(df, window_frames=5)
    assert g["g_n_frames"].iloc[0] == 1
    assert g["g_n_frames"].iloc[9] == 5
    # both IDs present in every 5-window
    assert (g["g_n_unique_ids"].iloc[4:] == 2).all()


def test_entropy_two_equal_ids():
    df = _table(40, ids=("a", "b"))
    g = build_global_features(df, window_frames=10)
    # at even indices the window has equal-ish counts of a and b
    e = g["g_id_entropy"].iloc[-1]
    assert e > 0.6
    assert g["g_id_entropy_norm"].iloc[-1] <= 1.0 + 1e-9
    assert g["g_dominant_id_frac"].iloc[-1] <= 1.0 + 1e-9


def test_single_id_has_zero_entropy_and_dominant_one():
    df = _table(20, ids=("a",))
    g = build_global_features(df, window_frames=5)
    assert (g["g_id_entropy"] == 0).all()
    assert (g["g_dominant_id_frac"] == 1.0).all()
    assert (g["g_id_concentration"] == 0).all()


def test_valid_ranges():
    df = _table(50, ids=("a", "b", "c"))
    g = build_global_features(df, window_frames=8)
    assert (g["g_n_unique_ids"] >= 1).all()
    assert (g["g_id_entropy"] >= -1e-9).all()
    assert (g["g_dominant_id_frac"] >= 0).all()
    assert (g["g_top3_id_frac"] <= 1.0 + 1e-9).all()
    for f in ALL_GLOBAL_FEATURES:
        finite = g[f].to_numpy(dtype=float)
        assert np.isfinite(finite).all()


def test_sparse_single_row_does_not_crash():
    df = _table(1, ids=("a",))
    g = build_global_features(df, window_frames=10)
    assert len(g) == 1
    assert g["g_n_unique_ids"].iloc[0] == 1


def test_no_future_leakage():
    df = _table(50, ids=("a", "b"))
    g1 = build_global_features(df, window_frames=10)
    df2 = df.copy()
    # modify only the final (future) row
    df2.loc[df2.index[-1], "byte_mean"] = 9999.0
    df2.loc[df2.index[-1], "dlc"] = 1
    g2 = build_global_features(df2, window_frames=10)
    for f in ALL_GLOBAL_FEATURES:
        pd.testing.assert_series_equal(
            g1[f].iloc[:-1], g2[f].iloc[:-1], check_names=False
        )


def test_families_partition_features():
    combined = BUS_FEATURES + ID_POPULATION_FEATURES + CROSS_ID_FEATURES
    assert combined == ALL_GLOBAL_FEATURES
    assert len(ALL_GLOBAL_FEATURES) == len(set(ALL_GLOBAL_FEATURES))
