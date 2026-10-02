"""Tests for hybrid score normalization and combination rules."""

from __future__ import annotations

import numpy as np
import pytest

from canguard.hybrid import (
    ScoreNormalizer,
    combine_scores,
    fit_score_normalizer,
    or_predict,
    threshold_at_fpr,
)


def test_score_normalizer():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    n = fit_score_normalizer(x)
    z = n.transform(x)
    assert np.isclose(z.mean(), 0.0, atol=1e-9)
    assert np.isclose(z.std(), 1.0, atol=1e-9)


def test_score_normalizer_constant_scale_guard():
    n = fit_score_normalizer(np.ones(10))
    assert n.scale == 1.0
    assert np.isfinite(n.transform(np.array([1.0, 2.0]))).all()


def test_provenance_guard():
    with pytest.raises(AssertionError):
        ScoreNormalizer(center=0.0, scale=1.0, provenance="test").transform(np.array([1.0]))


def test_or_rule():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 1, 0])
    assert or_predict(a, b).tolist() == [1, 1, 1, 0]


def test_max_rule():
    zl = np.array([0.0, 3.0, 1.0])
    zg = np.array([2.0, 1.0, 1.0])
    assert combine_scores(zl, zg, rule="max").tolist() == [2.0, 3.0, 1.0]


def test_weighted_rule_and_lambda_validation():
    zl = np.array([0.0, 2.0])
    zg = np.array([2.0, 0.0])
    out = combine_scores(zl, zg, rule="weighted", lam=0.75)
    assert np.allclose(out, [0.5, 1.5])
    with pytest.raises(ValueError):
        combine_scores(zl, zg, rule="weighted", lam=1.5)
    with pytest.raises(ValueError):
        combine_scores(zl, zg, rule="or")


def test_threshold_at_fpr_percentile():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=10000)
    thr = threshold_at_fpr(scores, 0.01)
    fpr = float((scores >= thr).mean())
    assert abs(fpr - 0.01) < 0.005
