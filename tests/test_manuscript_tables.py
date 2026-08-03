"""Tests for manuscript table formatting and ROAD protocol guards."""

from __future__ import annotations

import pandas as pd
import pytest

from canguard.exp.manuscript_tables import build_road_attack_summary, format_mean_std
from canguard.exp.metadata import (
    PROTOCOL_ROAD_LEGACY_FORBIDDEN,
    PROTOCOL_ROAD_PRE_INJECTION,
    assert_road_protocol_ok,
)


def test_n1_captures_suppress_std():
    """n=1 must not print a fabricated ±0.000 style variance."""
    s = format_mean_std(0.533, 0.0, n=1)
    assert "±" not in s
    assert "n=1" in s
    assert "n/a" in s.lower() or "std n/a" in s.lower()


def test_n_ge_2_prints_std():
    s = format_mean_std(0.5, 0.1, n=3)
    assert "±" in s
    assert "0.500" in s or "0.5" in s
    assert "n=3" in s


def test_assert_road_protocol_rejects_unversioned():
    df = pd.DataFrame({"f1": [0.1], "attack_type": ["x"]})
    with pytest.raises(RuntimeError, match="protocol"):
        assert_road_protocol_ok(df)


def test_assert_road_protocol_rejects_legacy():
    df = pd.DataFrame(
        {
            "f1": [0.1],
            "protocol_version": [PROTOCOL_ROAD_LEGACY_FORBIDDEN],
            "attack_type": ["x"],
        }
    )
    with pytest.raises(RuntimeError, match="non-corrected"):
        assert_road_protocol_ok(df)


def test_assert_road_protocol_accepts_pre_injection_v1():
    df = pd.DataFrame(
        {
            "f1": [0.5],
            "protocol": [PROTOCOL_ROAD_PRE_INJECTION],
            "capture": ["c1"],
            "attack_type": ["a"],
            "detector": ["isolation_forest"],
            "representation": ["residual"],
        }
    )
    assert_road_protocol_ok(df)


def test_road_summary_n_captures_and_fmt():
    df = pd.DataFrame(
        {
            "capture": ["a1", "a1"],
            "attack_type": ["correlated_signal_attack", "correlated_signal_attack"],
            "detector": ["isolation_forest", "hbos"],
            "representation": ["residual", "residual"],
            "protocol": [PROTOCOL_ROAD_PRE_INJECTION, PROTOCOL_ROAD_PRE_INJECTION],
            "f1": [0.5, 0.4],
            "recall": [1.0, 1.0],
            "precision": [0.3, 0.2],
            "roc_auc": [0.9, 0.85],
            "pr_auc": [0.7, 0.6],
            "fpr": [0.1, 0.15],
        }
    )
    summary = build_road_attack_summary(df, representation="residual")
    assert (summary["n_captures"] == 1).all()
    assert summary["f1_fmt"].str.contains("n=1").all()
    assert not summary["f1_fmt"].str.contains("±").any()


def test_sequence_autoencoder_registered():
    from canguard.detectors import create_detector, list_detectors

    assert "sequence_autoencoder" in list_detectors()
    det = create_detector("sequence_autoencoder", random_state=0, max_iter=5, max_train_samples=50)
    import numpy as np

    X = np.random.randn(80, 6)
    det.fit(X)
    s = det.score(X[:10])
    assert s.shape == (10,)
