"""ROAD dataset PIRD validation runner.

Runs the frozen PIRD pipeline (FeaturePipeline -> per-ID residuals ->
IsolationForest -> threshold -> evaluation) per ROAD attack capture.

Because ROAD captures are independent driving sessions, temporal structure
does not span captures. The faithful protocol here is per-capture:
  1. Parse the capture (loader already labels attack frames via metadata).
  2. Build per-ID windows with the unchanged FeaturePipeline.
  3. Calibration = normal windows in the pre-injection portion.
  4. Fit IF on additional normal windows; threshold on a held-out slice.
  5. Score the full capture; report metrics.

No feature, residual, or detector logic is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from canguard.data.road import RoadLoader, _parse_log
from canguard.detectors import IsolationForestDetector
from canguard.evaluation import (
    train_anomaly_detector,
)
from canguard.features import (
    BEHAVIORAL_FEATURES_V1 as FEATURES,
)
from canguard.features import (
    FeaturePipeline,
    fit_per_id_stats,
    temporal_split,
    transform_residuals,
)

_RESEARIAL = ["fpr", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"]


def _load_capture(road_dir: Path, capture_name: str) -> tuple[pd.DataFrame, dict]:
    """Load a single attack capture + its metadata, sorted by timestamp."""
    road_dir = Path(road_dir)
    loader = RoadLoader(road_dir)
    meta = loader._meta_all.get(capture_name, {})
    log = road_dir / "attacks" / f"{capture_name}.log"
    df = _parse_log(log, meta)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, meta


def _evaluate_capture(
    df_capture: pd.DataFrame,
    meta: dict,
    window_size: int = 30,
    fpr_target: float = 0.01,
    val_holdout_fraction: float = 0.2,
    max_frames: int = 100_000,
    random_state: int = 0,
) -> dict:
    """Run PIRD on one capture and return metrics + stored scores."""
    # Build per-ID windows (unchanged pipeline).
    pipe = FeaturePipeline(window_size=window_size)
    ft = pipe.process_dataframe(df_capture)

    if ft.empty:
        return {"error": "no windows produced"}

    res_cols = [c + "_res" for c in FEATURES]

    # Calibration: normal windows before the injection interval.
    # Map elapsed -> window timestamp is large absolute; injection interval is
    # elapsed, so we need capture-relative time. We use is_attack==0 for calib
    # (the frames NOT in the injection window) which is exactly the normals.
    normals = ft[ft["is_attack"] == 0].copy()
    if len(normals) < 50:
        return {"error": "too few normal windows"}

    # Order window table chronologically for temporal splits.
    ft = ft.sort_values("timestamp").reset_index(drop=True)

    # Per-id stats from normal windows.
    stats, global_stats = fit_per_id_stats(normals, FEATURES)
    # Residualize the whole capture.
    res = transform_residuals(ft, stats, global_stats, FEATURES)

    # Split residuals chronologically: calib/train/test as in HCRL spirit.
    calib, train_df, test_df = temporal_split(res, 0.4, 0.2, 0.4)

    det = IsolationForestDetector(n_estimators=200, random_state=random_state)
    out = train_anomaly_detector(
        det,
        train_df,
        test_df,
        res_cols,
        val_holdout_fraction=val_holdout_fraction,
        fpr_target=fpr_target,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ROAD PIRD validation runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    road_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = RoadLoader(road_dir)
    attack_meta = loader._meta_all
    # Only attack captures (a .log present under road/attacks/) are evaluated.
    attack_logs = {p.stem for p in loader._collect_logs("attacks")}
    capture_names = sorted(name for name in attack_meta if name in attack_logs)

    results = {}
    for name in capture_names:
        df_cap, meta = _load_capture(road_dir, name)
        # Skip accelerator-type captures (unlabeled) and masquerade duplicates.
        atype = meta.get("injection_id")
        if atype in (None,) and meta.get("injection_interval") is None:
            continue
        if name.endswith("_masquerade"):
            continue
        out = _evaluate_capture(
            df_cap,
            meta,
            window_size=cfg["window_size"],
            fpr_target=cfg["evaluation"]["fpr_target"],
            val_holdout_fraction=cfg["evaluation"]["val_holdout_fraction"],
            max_frames=cfg.get("max_frames_per_capture", 100_000),
            random_state=cfg.get("random_state", 0),
        )
        results[name] = {k: out.get(k) for k in _RESEARIAL}
        print(f"[{name}] F1={out.get('f1')} Recall={out.get('recall')} " f"FPR={out.get('fpr')}")

    out_path = out_dir / "road_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
