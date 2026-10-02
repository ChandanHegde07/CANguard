
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(
    ft: pd.DataFrame,
    calib_frac: float = 0.4,
    train_frac: float = 0.2,
    test_frac: float = 0.4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
