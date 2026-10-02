"""Unit tests for detection-boundary analysis."""

from __future__ import annotations

import numpy as np
import pytest

from canguard.evaluation.boundary import (
    crossing_severity,
    detection_probability,
    summarize_boundary,
    wilson_interval,
)


def test_wilson_interval_basic():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0
    assert 0.0 < hi < 0.05
    lo, hi = wilson_interval(100, 100)
    assert hi == pytest.approx(1.0)
    assert 0.95 < lo < 1.0
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert hi - lo == pytest.approx(0.196, abs=0.01)


def test_detection_probability_counts():
    flags = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    out = detection_probability(flags)
    assert out["n"] == 8
    assert out["n_detected"] == 3
    assert out["detection_probability"] == pytest.approx(3 / 8)
    assert out["detection_prob_ci_low"] < 0.375 < out["detection_prob_ci_high"]


def test_crossing_floor_and_ceiling():
    sev = np.array([0.25, 1.0, 3.0])
    assert crossing_severity(sev, np.array([0.6, 0.8, 1.0]), 0.5) == (None, "floor_exceeded")
    assert crossing_severity(sev, np.array([0.0, 0.1, 0.2]), 0.5) == (None, "ceiling_not_reached")


def test_crossing_interpolation():
    sev = np.array([0.25, 1.0, 3.0])
    probs = np.array([0.2, 0.6, 0.9])
    # 0.5 lies halfway between alpha 0.25 (0.2) and 1.0 (0.6)
    val, reason = crossing_severity(sev, probs, 0.5)
    assert reason == ""
    assert val == pytest.approx(0.25 + 0.75 * (0.3 / 0.4))


def test_summarize_boundary_reports_reasons():
    sev = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
    probs = np.array([0.05, 0.2, 0.55, 0.85, 0.98])
    out = summarize_boundary(sev, probs)
    assert out["alpha_50"] == pytest.approx(0.5 + 0.5 * (0.3 / 0.35))  # 0.9286
    assert out["alpha_90"] == pytest.approx(2.0 + 1.0 * (0.05 / 0.13))
    assert out["alpha_50_reason"] == ""
    assert out["alpha_90_reason"] == ""
