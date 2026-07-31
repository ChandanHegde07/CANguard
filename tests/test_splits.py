"""Tests for temporal split logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canguard.features.splits import temporal_split


def _seq_table(n=1000):
    return pd.DataFrame({"timestamp": np.arange(n), "can_id": ["0316"] * n, "is_attack": [0] * n})


def test_temporal_split_contiguity_and_order():
    ft = _seq_table(1000)
    calib, train, test = temporal_split(ft, 0.4, 0.2, 0.4)
    assert len(calib) == 400
    assert len(train) == 200
    assert len(test) == 400
    assert calib["timestamp"].iloc[-1] < train["timestamp"].iloc[0]
    assert train["timestamp"].iloc[-1] < test["timestamp"].iloc[0]


def test_temporal_split_fractions_must_sum_to_one():
    ft = _seq_table()
    with pytest.raises(ValueError, match="must equal 1.0"):
        temporal_split(ft, 0.5, 0.3)  # only 0.8 total
