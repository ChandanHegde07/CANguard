"""Correct ROAD per-capture protocol for residualization studies.

Key rules (fix audit B1–B4):
  * Residual μ/σ fitted only on **pre-injection** windows (elapsed < start).
  * Detector trained only on pre-injection normals.
  * Threshold calibrated on a holdout of pre-injection normals.
  * Test = windows with elapsed >= injection start (attack period + after).
  * Raw and residual use the **same** train/test window indices.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from canguard.data.road import RoadLoader, _parse_log
from canguard.features import (
    BEHAVIORAL_FEATURES_V1,
    FeaturePipeline,
    fit_per_id_stats,
    transform_residuals,
)
from canguard.exp.cache import FeatureCache

logger = logging.getLogger("canguard")


def resolve_road_root(data_dir: str | Path) -> Path:
    """Accept ``road/``, ``road/road/``, or a path that already contains attacks/."""
    p = Path(data_dir)
    candidates = [p, p / "road", Path("road/road"), Path("road")]
    for c in candidates:
        if (c / "attacks").is_dir() and (c / "attacks" / "capture_metadata.json").exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Could not resolve ROAD root from {data_dir!r}. "
        "Expected a directory containing attacks/capture_metadata.json."
    )


def list_eval_captures(
    road_root: Path,
    skip_masquerade: bool = True,
    skip_unlabeled: bool = True,
    per_type: int | None = None,
) -> list[tuple[str, dict]]:
    """Return (capture_name, meta) pairs eligible for evaluation."""
    loader = RoadLoader(road_root)
    attack_logs = {p.stem for p in loader._collect_logs("attacks")}
    items: list[tuple[str, dict]] = []
    for name in sorted(loader._meta_all):
        if name not in attack_logs:
            continue
        meta = loader._meta_all[name]
        if skip_masquerade and name.endswith("_masquerade"):
            continue
        if skip_unlabeled and meta.get("injection_interval") is None:
            continue
        items.append((name, meta))

    if per_type is None:
        return items

    # Keep first ``per_type`` captures per attack-type family.
    from collections import defaultdict

    buckets: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for name, meta in items:
        # e.g. max_speedometer_attack_1 -> max_speedometer_attack
        parts = name.split("_")
        if parts and parts[-1].isdigit():
            atype = "_".join(parts[:-1])
        else:
            atype = name
        if len(buckets[atype]) < per_type:
            buckets[atype].append((name, meta))
    out: list[tuple[str, dict]] = []
    for atype in sorted(buckets):
        out.extend(buckets[atype])
    return out


def load_capture_frames(
    road_root: Path,
    capture_name: str,
    meta: dict,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """Load one attack capture; optionally truncate while covering injection."""
    log = road_root / "attacks" / f"{capture_name}.log"
    df = _parse_log(log, meta)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "elapsed" not in df.columns:
        df["elapsed"] = df["timestamp"] - df["timestamp"].iloc[0]

    if max_frames is None or len(df) <= max_frames:
        return df

    interval = meta.get("injection_interval")
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        start, end = float(interval[0]), float(interval[1])
        pre = df[df["elapsed"] < start]
        # Always prioritize injection region (+ short post buffer).
        inj = df[(df["elapsed"] >= start) & (df["elapsed"] <= end + 5.0)]
        if len(inj) == 0:
            # Fallback: keep the latest max_frames rows (most likely to include late injections).
            return df.iloc[-max_frames:].reset_index(drop=True)
        if len(inj) >= max_frames:
            return inj.iloc[:max_frames].reset_index(drop=True)
        n_pre = max_frames - len(inj)
        pre_part = pre.iloc[-n_pre:] if len(pre) > n_pre else pre
        return pd.concat([pre_part, inj], ignore_index=True)

    return df.iloc[:max_frames].reset_index(drop=True)


def build_road_window_table(
    df_frames: pd.DataFrame,
    window_size: int = 30,
    cache: FeatureCache | None = None,
    cache_key: dict | None = None,
) -> pd.DataFrame:
    """Build per-ID windows; attach elapsed relative to capture start."""
    if cache is not None and cache_key is not None:
        hit = cache.load_df(cache_key)
        if hit is not None:
            return hit

    if df_frames.empty:
        return pd.DataFrame()

    # Capture origin time: elapsed must stay relative to capture start, not the
    # first *loaded* frame (max_frames may drop the true start).
    if "elapsed" in df_frames.columns:
        t0 = float(df_frames["timestamp"].iloc[0] - df_frames["elapsed"].iloc[0])
    else:
        t0 = float(df_frames["timestamp"].iloc[0])

    pipe = FeaturePipeline(window_size=window_size)
    ft = pipe.process_dataframe(df_frames)
    if ft.empty:
        return ft
    ft = ft.sort_values("timestamp").reset_index(drop=True)
    ft["elapsed"] = ft["timestamp"] - t0

    if cache is not None and cache_key is not None:
        cache.save_df(cache_key, ft)
    return ft


def build_road_raw_residual_splits(
    feature_table: pd.DataFrame,
    injection_interval: list | tuple,
    feature_cols: list[str] | None = None,
    min_pre_windows: int = 50,
) -> dict[str, Any]:
    """Pre-injection train / post-start test splits for raw and residual.

    Parameters
    ----------
    feature_table
        Window table with ``elapsed`` and ``is_attack``.
    injection_interval
        ``[start_sec, end_sec]`` in capture-elapsed seconds.
    """
    feature_cols = list(feature_cols or BEHAVIORAL_FEATURES_V1)
    start = float(injection_interval[0])

    ft = feature_table.sort_values("timestamp").reset_index(drop=True)
    if "elapsed" not in ft.columns:
        raise ValueError("feature_table must include elapsed seconds")

    pre = ft[ft["elapsed"] < start].copy()
    test = ft[ft["elapsed"] >= start].copy()

    if len(pre) < min_pre_windows:
        return {
            "error": f"too few pre-injection windows ({len(pre)} < {min_pre_windows})",
            "n_pre": len(pre),
            "n_test": len(test),
        }
    if len(test) < 10:
        return {
            "error": f"too few post-start windows ({len(test)})",
            "n_pre": len(pre),
            "n_test": len(test),
        }
    if int(test["is_attack"].sum()) == 0:
        return {
            "error": "no attack windows in test segment",
            "n_pre": len(pre),
            "n_test": len(test),
        }

    # Residual stats: pre-injection only (all should be normals).
    stats, gstats = fit_per_id_stats(pre, feature_cols)
    residual_cols = [c + "_res" for c in feature_cols]

    res_pre = transform_residuals(pre, stats, gstats, feature_cols)
    res_test = transform_residuals(test, stats, gstats, feature_cols)

    return {
        "raw": {"train": pre, "test": test},
        "residual": {"train": res_pre, "test": res_test},
        "feature_cols": feature_cols,
        "residual_cols": residual_cols,
        "per_id_stats_n": len(stats),
        "n_pre": len(pre),
        "n_test": len(test),
        "n_test_attack": int(test["is_attack"].sum()),
        "injection_start": start,
        "injection_end": float(injection_interval[1]),
    }


def prepare_road_capture(
    road_root: Path,
    capture_name: str,
    meta: dict,
    window_size: int = 30,
    max_frames: int | None = 100_000,
    cache: FeatureCache | None = None,
) -> dict[str, Any]:
    """End-to-end: load capture → windows → raw/residual splits."""
    interval = meta.get("injection_interval")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return {"error": "missing injection_interval", "capture": capture_name}

    df = load_capture_frames(road_root, capture_name, meta, max_frames=max_frames)
    if df.empty:
        return {"error": "empty capture", "capture": capture_name}

    cache_key = None
    if cache is not None:
        log = road_root / "attacks" / f"{capture_name}.log"
        cache_key = {
            "stage": "road_windows",
            "capture": capture_name,
            "window_size": window_size,
            "max_frames": max_frames,
            "n_frames": len(df),
            "mtime": log.stat().st_mtime if log.exists() else 0,
            # Bump when frame-selection logic changes so stale caches are not reused.
            "protocol": "pre_injection_v2_inj_priority",
        }

    ft = build_road_window_table(df, window_size=window_size, cache=cache, cache_key=cache_key)
    if ft.empty:
        return {"error": "no windows", "capture": capture_name}

    splits = build_road_raw_residual_splits(ft, interval)
    splits["capture"] = capture_name
    splits["n_frames"] = len(df)
    splits["attack_type"] = _attack_type(capture_name)
    return splits


def _attack_type(capture_name: str) -> str:
    parts = capture_name.split("_")
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return capture_name
