
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from canguard.data import get_loader
from canguard.features import (
    BEHAVIORAL_FEATURES_V1,
    FeaturePipeline,
    fit_known_ids_on_normal_prefix,
    fit_per_id_stats,
    temporal_split,
    transform_residuals,
)
from canguard.exp.cache import FeatureCache

logger = logging.getLogger("canguard")


def resolve_data_path(data_dir: str | Path, dataset_name: str) -> Path:
    """Resolve HCRL CSV path: prefer data_dir/NAME_dataset.csv."""
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"{dataset_name}_dataset.csv",
        data_dir / f"{dataset_name}.csv",
        Path("HCRL Car-Hacking") / f"{dataset_name}_dataset.csv",
        Path("data") / f"{dataset_name}_dataset.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find dataset '{dataset_name}'. Tried: {[str(c) for c in candidates]}"
    )


def build_window_table(
    csv_path: Path,
    window_size: int,
    sample_size: int | None,
    cache: FeatureCache | None = None,
) -> pd.DataFrame:
    """Load frames and build per-ID window feature table (cached)."""
    fp = {
        "stage": "windows",
        "path": str(csv_path.resolve()),
        "window_size": window_size,
        "sample_size": sample_size,
        "mtime": csv_path.stat().st_mtime,
    }
    if cache is not None:
        hit = cache.load_df(fp)
        if hit is not None:
            return hit

    logger.info("Loading %s (sample_size=%s)", csv_path, sample_size)
    df = get_loader("hcrl", csv_path).load(sample_size=sample_size)
    known = fit_known_ids_on_normal_prefix(df)
    pipe = FeaturePipeline(window_size=window_size, known_ids=known)
    ft = pipe.process_dataframe(df)
    logger.info("  %s windows from %s frames", len(ft), len(df))
    if cache is not None:
        cache.save_df(fp, ft)
    return ft


def build_raw_and_residual_splits(
    feature_table: pd.DataFrame,
    feature_cols: list[str] | None = None,
    calib_frac: float = 0.4,
    train_frac: float = 0.2,
    test_frac: float = 0.4,
) -> dict[str, Any]:

    feature_cols = list(feature_cols or BEHAVIORAL_FEATURES_V1)
    calib, train, test = temporal_split(feature_table, calib_frac, train_frac, test_frac)
    stats, gstats = fit_per_id_stats(calib, feature_cols)
    residual_cols = [c + "_res" for c in feature_cols]
    return {
        "raw": {"calib": calib, "train": train, "test": test},
        "residual": {
            "calib": transform_residuals(calib, stats, gstats, feature_cols),
            "train": transform_residuals(train, stats, gstats, feature_cols),
            "test": transform_residuals(test, stats, gstats, feature_cols),
        },
        "feature_cols": feature_cols,
        "residual_cols": residual_cols,
        "per_id_stats_n": len(stats),
    }
