"""CLI runner: train/evaluate a PIRD detector from a YAML config.

Usage:
    python -m experiments.runners.train_detector --config experiments/configs/hcrl.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from canguard.data import get_loader
from canguard.detectors import IsolationForestDetector
from canguard.evaluation import train_anomaly_detector
from canguard.features import (
    BEHAVIORAL_FEATURES_V1,
    FeaturePipeline,
    fit_known_ids_on_normal_prefix,
    fit_per_id_stats,
    temporal_split,
    transform_residuals,
)

_SERIALIZABLE_KEYS = (
    "fpr",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "threshold",
    "n_train_normals",
)


def _resolve_feature_cols(config: dict) -> list[str]:
    """Return the feature column set selected by the config (v1 full by default)."""
    variant_groups = config.get("variant_feature_groups")
    if variant_groups:
        from canguard.features.groups import (
            GROUP_DLC,
            GROUP_FLAT_BYTE,
            GROUP_IAT,
            GROUP_OTHER,
        )

        group_map = {
            "iat": GROUP_IAT,
            "flat_byte": GROUP_FLAT_BYTE,
            "dlc": GROUP_DLC,
            "other": GROUP_OTHER,
        }
        cols: list[str] = []
        for g in variant_groups:
            cols.extend(group_map[g])
        return cols
    return config.get("feature_cols") or list(BEHAVIORAL_FEATURES_V1)


def run_experiment(config: dict) -> dict:
    """Execute the PIRD baseline experiment described by ``config``.

    Returns a JSON-serializable results dict keyed by dataset name.
    """
    data_dir = Path(config["data_dir"])
    sample_size = config.get("sample_size")
    ws = config.get("window_size", 30)
    names = config["datasets"]
    split = config["split"]
    eval_cfg = config["evaluation"]
    det_cfg = config["detector"]
    feature_cols = _resolve_feature_cols(config)

    results: dict = {}
    for name in names:
        df = get_loader(config["dataset"], data_dir / f"{name}_dataset.csv").load(
            sample_size=sample_size
        )
        known = fit_known_ids_on_normal_prefix(df)
        pipe = FeaturePipeline(window_size=ws, known_ids=known)
        ft = pipe.process_dataframe(df)
        pipe.reset()

        calib, train, test = temporal_split(
            ft, split["calib_frac"], split["train_frac"], split["test_frac"]
        )
        stats, global_stats = fit_per_id_stats(calib, feature_cols)
        res_cols = [c + "_res" for c in feature_cols]
        res_train = transform_residuals(train, stats, global_stats, feature_cols)
        res_test = transform_residuals(test, stats, global_stats, feature_cols)

        det = IsolationForestDetector(
            n_estimators=det_cfg.get("n_estimators", 200),
            random_state=det_cfg.get("random_state", 0),
        )
        out = train_anomaly_detector(
            det,
            res_train,
            res_test,
            res_cols,
            val_holdout_fraction=eval_cfg.get("val_holdout_fraction", 0.2),
            fpr_target=eval_cfg.get("fpr_target", 0.01),
        )
        serializable = {
            k: (out[k] if isinstance(out[k], (int, float, str)) else None)
            for k in _SERIALIZABLE_KEYS
            if k in out
        }
        results[name] = serializable
        print(
            f"[{name}] F1={serializable['f1']:.3f} "
            f"Recall={serializable['recall']:.3f} FPR={serializable['fpr']:.4f}"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PIRD experiment runner")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument("--output", default=None, help="Override output JSON path")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    results = run_experiment(config)

    out = Path(args.output) if args.output else Path(config["output_dir"]) / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    main()
