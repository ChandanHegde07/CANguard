"""Tests for the global distribution-shift diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canguard.evaluation.distribution_shift import (
    association_with_flags,
    distribution_shift,
    family_shift_summary,
    feature_shift_table,
    standardized_mean_difference,
)

FAMILIES = {"bus": ["a", "b"], "id_population": ["c"], "cross_id": ["d"]}


def test_smd_known():
    assert standardized_mean_difference([0, 1, 2], [1, 2, 3]) == pytest.approx(1.0)
    assert standardized_mean_difference([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_ks_and_wasserstein():
    d = distribution_shift([0, 1, 2, 3], [0, 1, 2, 3])
    assert d["ks"] == pytest.approx(0.0)
    assert d["wasserstein"] == pytest.approx(0.0)
    d2 = distribution_shift([0, 0, 0, 0], [5, 5, 5, 5])
    assert d2["ks"] == pytest.approx(1.0)
    assert d2["wasserstein"] == pytest.approx(5.0)
    assert np.isinf(d2["smd"]) and d2["smd"] > 0  # zero variance, shifted mean


def test_distribution_shift_keys_and_quantiles():
    d = distribution_shift(np.arange(100), np.arange(100) + 10)
    for key in ("cal_mean", "test_mean", "cal_std", "test_std", "smd", "ks",
                "wasserstein", "normalized_wasserstein", "cal_q50", "test_q50"):
        assert key in d
    assert d["test_mean"] - d["cal_mean"] == pytest.approx(10.0)


def _frames(seed=0, n=80):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n) + 3.0,   # shifted
        "c": rng.normal(0, 1, n),
        "d": rng.normal(0, 1, n) + 0.1,
    })


def test_feature_shift_ranking_and_family():
    cal = _frames(0)
    test = _frames(1)
    table = feature_shift_table(cal, test, ["a", "b", "c", "d"], FAMILIES)
    assert list(table.columns)[:2] == ["feature", "family"]
    # ranking is by |SMD|, descending
    assert (table["abs_smd"].diff().dropna() <= 1e-9).all()
    assert set(table["family"]) == {"bus", "id_population", "cross_id"}


def test_feature_shift_deterministic():
    cal = _frames(0)
    test = _frames(1)
    a = feature_shift_table(cal, test, ["a", "b", "c", "d"], FAMILIES)
    b = feature_shift_table(cal, test, ["a", "b", "c", "d"], FAMILIES)
    pd.testing.assert_frame_equal(a, b)


def test_family_shift_summary():
    cal = _frames(0)
    test = _frames(1)
    table = feature_shift_table(cal, test, ["a", "b", "c", "d"], FAMILIES)
    fam = family_shift_summary(table)
    assert {"family", "mean_abs_smd", "mean_ks", "mean_normalized_wasserstein"} <= set(fam.columns)
    # 'b' is strongly shifted and belongs to bus -> bus should rank high.
    assert fam.iloc[0]["family"] == "bus"


def test_association_uses_normal_windows_only():
    cal = _frames(0)
    test = _frames(1)
    n = len(test)
    flags = np.zeros(n, dtype=bool)
    flags[: n // 2] = True
    scores = np.linspace(0, 1, n)
    assoc = association_with_flags(cal, test, flags, scores, ["a", "b", "c", "d"], FAMILIES)
    assert (assoc["n_flagged"] + assoc["n_normal"] == n).all()
    assert "spearman_deviation_vs_score" in assoc.columns


def test_ranking_reproducible_across_calls():
    cal = _frames(3)
    test = _frames(4)
    r1 = feature_shift_table(cal, test, ["a", "b", "c", "d"], FAMILIES)["feature"].tolist()
    r2 = feature_shift_table(cal, test, ["a", "b", "c", "d"], FAMILIES)["feature"].tolist()
    assert r1 == r2
