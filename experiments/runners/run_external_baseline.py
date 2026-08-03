"""Run external sequence-autoencoder baseline under the same CANguard protocol.

Comparison point only (CANet / sequence-reconstruction inspired). Not a SOTA claim.
Uses the same splits, residualization, and FPR-target thresholding as Phase A/B.

Usage:
    python -m experiments.runners.run_external_baseline --config experiments/configs/external_baseline.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.detectors import create_detector
from canguard.evaluation import train_anomaly_detector
from canguard.exp import FeatureCache, load_config, save_csv, set_global_seed, setup_logging
from canguard.exp.matrix import (
    build_raw_and_residual_splits,
    build_window_table,
    resolve_data_path,
)
from canguard.exp.metadata import (
    PHASE_WORKSHOP,
    PROTOCOL_HCRL_TEMPORAL,
    PROTOCOL_ROAD_PRE_INJECTION,
    tag_dataframe,
)
from canguard.exp.road_protocol import (
    list_eval_captures,
    prepare_road_capture,
    resolve_road_root,
)

logger = logging.getLogger("canguard")


def main() -> None:
    parser = argparse.ArgumentParser(description="External sequence-AE baseline")
    parser.add_argument("--config", default="experiments/configs/external_baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)
    tables = Path(cfg.get("tables_dir", "tables"))
    tables.mkdir(parents=True, exist_ok=True)
    setup_logging()
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase_c"))

    det_name = cfg.get("detector", "sequence_autoencoder")
    params = dict(cfg.get("detector_params", {}).get(det_name, {}))
    params["random_state"] = seed
    eval_cfg = cfg["evaluation"]
    rows = []

    # --- HCRL ---
    h = cfg["hcrl"]
    for name in h["datasets"]:
        path = resolve_data_path(h["data_dir"], name)
        ft = build_window_table(path, int(h["window_size"]), h.get("sample_size"), cache=cache)
        mat = build_raw_and_residual_splits(ft, **h["split"])
        for rep in cfg.get("representations", ["raw", "residual"]):
            train = mat[rep]["train"]
            test = mat[rep]["test"]
            cols = mat["feature_cols"] if rep == "raw" else mat["residual_cols"]
            logger.info("external HCRL %s %s", name, rep)
            det = create_detector(det_name, **params)
            out = train_anomaly_detector(
                det,
                train,
                test,
                cols,
                val_holdout_fraction=eval_cfg["val_holdout_fraction"],
                fpr_target=eval_cfg["fpr_target"],
                measure_resources=True,
            )
            rows.append(
                {
                    "corpus": "hcrl",
                    "dataset": name,
                    "capture": name,
                    "attack_type": name,
                    "representation": rep,
                    "detector": det_name,
                    "seed": seed,
                    "precision": out["precision"],
                    "recall": out["recall"],
                    "f1": out["f1"],
                    "roc_auc": out["roc_auc"],
                    "pr_auc": out["pr_auc"],
                    "fpr": out["fpr"],
                    "train_seconds": out.get("train_seconds"),
                    "score_seconds": out.get("score_seconds"),
                    "model_bytes": out.get("model_bytes"),
                    "comparison_role": "external_baseline",
                    "input_note": "same_14d_window_features_seq_len_5_not_raw_can_bytes",
                }
            )
            logger.info("  F1=%.3f ROC=%.3f", out["f1"], out["roc_auc"])

    # --- ROAD ---
    rcfg = cfg["road"]
    root = resolve_road_root(rcfg["data_dir"])
    cs = rcfg.get("capture_sample", {})
    captures = list_eval_captures(
        root,
        skip_masquerade=cs.get("skip_masquerade", True),
        skip_unlabeled=cs.get("skip_unlabeled", True),
        per_type=cs.get("per_type"),
    )
    for cap, meta in captures:
        prep = prepare_road_capture(
            root,
            cap,
            meta,
            window_size=int(rcfg["window_size"]),
            max_frames=rcfg.get("max_frames_per_capture"),
            cache=cache,
        )
        if prep.get("error"):
            logger.warning("skip %s: %s", cap, prep["error"])
            continue
        for rep in cfg.get("representations", ["raw", "residual"]):
            train = prep[rep]["train"]
            test = prep[rep]["test"]
            cols = prep["feature_cols"] if rep == "raw" else prep["residual_cols"]
            logger.info("external ROAD %s %s", cap, rep)
            det = create_detector(det_name, **params)
            out = train_anomaly_detector(
                det,
                train,
                test,
                cols,
                val_holdout_fraction=eval_cfg["val_holdout_fraction"],
                fpr_target=eval_cfg["fpr_target"],
                measure_resources=True,
            )
            rows.append(
                {
                    "corpus": "road",
                    "dataset": prep.get("attack_type", cap),
                    "capture": cap,
                    "attack_type": prep.get("attack_type", cap),
                    "representation": rep,
                    "detector": det_name,
                    "seed": seed,
                    "precision": out["precision"],
                    "recall": out["recall"],
                    "f1": out["f1"],
                    "roc_auc": out["roc_auc"],
                    "pr_auc": out["pr_auc"],
                    "fpr": out["fpr"],
                    "train_seconds": out.get("train_seconds"),
                    "score_seconds": out.get("score_seconds"),
                    "model_bytes": out.get("model_bytes"),
                    "comparison_role": "external_baseline",
                    "input_note": "same_14d_window_features_seq_len_5_not_raw_can_bytes",
                    "n_pre": prep.get("n_pre"),
                    "n_test": prep.get("n_test"),
                    "n_test_attack": prep.get("n_test_attack"),
                }
            )
            logger.info("  F1=%.3f ROC=%.3f", out["f1"], out["roc_auc"])

    import pandas as pd

    df = pd.DataFrame(rows)
    # Tag protocol per corpus
    hcrl_mask = df["corpus"] == "hcrl"
    road_mask = df["corpus"] == "road"
    df.loc[hcrl_mask, "protocol_version"] = PROTOCOL_HCRL_TEMPORAL
    df.loc[road_mask, "protocol_version"] = PROTOCOL_ROAD_PRE_INJECTION
    df["phase"] = PHASE_WORKSHOP
    if road_mask.any():
        nmap = (
            df.loc[road_mask]
            .groupby("attack_type")["capture"]
            .nunique()
            .to_dict()
        )
        df.loc[road_mask, "n_captures"] = df.loc[road_mask, "attack_type"].map(nmap)

    out_path = tables / "external_baseline_results.csv"
    save_csv(out_path, df)
    save_csv(Path(cfg.get("output_dir", "experiments")) / "external_baseline" / "results.csv", df)
    print(f"\nWrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
