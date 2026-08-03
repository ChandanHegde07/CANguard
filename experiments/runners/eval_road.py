"""ROAD validation runner (corrected pre-injection protocol).

Usage:
    python -m experiments.runners.eval_road --config experiments/configs/road.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from canguard.detectors import IsolationForestDetector
from canguard.evaluation import train_anomaly_detector
from canguard.exp.road_protocol import (
    list_eval_captures,
    prepare_road_capture,
    resolve_road_root,
)

_RESEARIAL = ["fpr", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"]


def main() -> None:
    parser = argparse.ArgumentParser(description="ROAD PIRD validation (pre-injection protocol)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    road_root = resolve_road_root(cfg["data_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    captures = list_eval_captures(
        road_root,
        skip_masquerade=True,
        skip_unlabeled=True,
        per_type=cfg.get("capture_sample", {}).get("per_type"),
    )

    results = {}
    for name, meta in captures:
        prep = prepare_road_capture(
            road_root,
            name,
            meta,
            window_size=cfg.get("window_size", 30),
            max_frames=cfg.get("max_frames_per_capture"),
        )
        if prep.get("error"):
            results[name] = {"error": prep["error"]}
            print(f"[{name}] SKIP {prep['error']}")
            continue
        det = IsolationForestDetector(
            n_estimators=200,
            random_state=cfg.get("random_state", 0),
            n_jobs=1,
        )
        out = train_anomaly_detector(
            det,
            prep["residual"]["train"],
            prep["residual"]["test"],
            prep["residual_cols"],
            val_holdout_fraction=cfg["evaluation"]["val_holdout_fraction"],
            fpr_target=cfg["evaluation"]["fpr_target"],
        )
        results[name] = {k: out.get(k) for k in _RESEARIAL}
        results[name]["protocol"] = "pre_injection_v1"
        print(
            f"[{name}] F1={out.get('f1'):.3f} Recall={out.get('recall'):.3f} "
            f"FPR={out.get('fpr'):.4f} ROC={out.get('roc_auc'):.3f}"
        )

    out_path = out_dir / "road_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
