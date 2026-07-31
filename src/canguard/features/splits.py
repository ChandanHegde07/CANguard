"""Chronological train/calib/test splits for window feature tables.

Temporal splits only -- never shuffle windows across time. This preserves the
time-ordering assumption of the residual pipeline and the anomaly detectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(
    ft: pd.DataFrame,
    calib_frac: float = 0.4,
    train_frac: float = 0.2,
    test_frac: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a window feature table chronologically into calib/train/test.

    The full table is split by row index into three contiguous, time-ordered
    segments. The three fractions must sum to 1.0 (within ``np.isclose``
    tolerance).

    Parameters
    ----------
    ft : pd.DataFrame
        Feature table whose row order is chronological (as produced by
        ``FeaturePipeline.process_dataframe``).
    calib_frac : float
        Fraction of rows reserved as the calibration segment (default 0.4).
    train_frac : float
        Fraction of rows reserved as the training segment (default 0.2).
    test_frac : float
        Fraction of rows reserved as the test segment (default 0.4).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(calib, train, test)`` contiguous copies in time order.
    """
    if not np.isclose(calib_frac + train_frac + test_frac, 1.0):
        raise ValueError(
            f"calib_frac + train_frac + test_frac must equal 1.0, "
            f"got {calib_frac + train_frac + test_frac}"
        )
    n = len(ft)
    calib_end = int(n * calib_frac)
    train_end = calib_end + int(n * train_frac)
    return (
        ft.iloc[:calib_end].copy(),
        ft.iloc[calib_end:train_end].copy(),
        ft.iloc[train_end:].copy(),
    )
