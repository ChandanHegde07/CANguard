"""Phase B ROAD runner: raw vs residual × multi-detector (pre-injection protocol).

Usage:
    python -m experiments.runners.run_phase_b_road --config experiments/configs/phase_b_road.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.detectors import create_detector
from canguard.evaluation import bootstrap_metrics, flatten_bootstrap, train_anomaly_detector
from canguard.exp import (
    ExperimentRun,
    FeatureCache,
    load_config,
    save_csv,
    save_json,
    set_global_seed,
    setup_logging,
)
from canguard.exp.road_protocol import (
    list_eval_captures,
    prepare_road_capture,
    resolve_road_root,
)

logger = logging.getLogger("canguard")


def _hypothesis_verdict(rep_df: pd.DataFrame) -> dict:
    """Same decision rule as Phase A, cells = (capture, detector)."""
    pivot_f1 = rep_df.pivot_table(
        index=["capture", "detector"], columns="representation", values="f1", aggfunc="first"
    )
    pivot_roc = rep_df.pivot_table(
        index=["capture", "detector"], columns="representation", values="roc_auc", aggfunc="first"
    )
    if "raw" not in pivot_f1.columns or "residual" not in pivot_f1.columns:
        return {"verdict": "incomplete", "reason": "missing raw or residual"}

    d_f1 = (pivot_f1["residual"] - pivot_f1["raw"]).dropna()
    d_roc = (pivot_roc["residual"] - pivot_roc["raw"]).dropna()
    n = len(d_f1)
    n_f1_pos = int((d_f1 > 0).sum())
    n_roc_pos = int((d_roc > 0).sum())
    frac_f1 = n_f1_pos / n if n else 0.0
    frac_roc = n_roc_pos / n if n else 0.0
    median_df1 = float(d_f1.median()) if n else float("nan")
    median_droc = float(d_roc.median()) if n else float("nan")

    # Targeted AID vs fuzzing split
    captures = d_f1.index.get_level_values("capture")
    is_fuzz = captures.str.contains("fuzzing")
    fuzz_frac = float((d_f1[is_fuzz] > 0).mean()) if is_fuzz.any() else float("nan")
    targ_frac = float((d_f1[~is_fuzz] > 0).mean()) if (~is_fuzz).any() else float("nan")

    if frac_f1 >= 0.70 and median_df1 > 0 and frac_roc >= 0.60:
        verdict = "A_supported"
    elif frac_f1 >= 0.50 or (median_df1 > 0 and (targ_frac == targ_frac and targ_frac >= 0.75)):
        verdict = "B_partially_supported"
    else:
        verdict = "C_rejected"

    per_cell = []
    for (cap, det), df1 in d_f1.items():
        per_cell.append(
            {
                "capture": cap,
                "detector": det,
                "delta_f1": float(df1),
                "delta_roc_auc": float(d_roc.loc[(cap, det)]) if (cap, det) in d_roc.index else float("nan"),
                "residual_helps_f1": bool(df1 > 0),
            }
        )

    return {
        "verdict": verdict,
        "n_cells": n,
        "frac_f1_positive": frac_f1,
        "frac_roc_positive": frac_roc,
        "median_delta_f1": median_df1,
        "median_delta_roc_auc": median_droc,
        "targeted_frac_f1_positive": targ_frac,
        "fuzzing_frac_f1_positive": fuzz_frac,
        "n_f1_positive": n_f1_pos,
        "n_roc_positive": n_roc_pos,
        "per_cell": per_cell,
        "protocol": "pre_injection_v1",
        "decision_rule": {
            "A_supported": "frac_f1>=0.70 & median_dF1>0 & frac_roc>=0.60",
            "B_partially_supported": "frac_f1>=0.50 OR (median_dF1>0 & targeted_frac>=0.75)",
            "C_rejected": "otherwise",
        },
    }


def _plot_bars(rep_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Aggregate mean F1 by attack_type × representation × detector
    if rep_df.empty:
        return
    for metric in ("f1", "roc_auc"):
        types = sorted(rep_df["attack_type"].unique())
        detectors = sorted(rep_df["detector"].unique())
        fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 4), sharey=True)
        if len(types) == 1:
            axes = [axes]
        x = np.arange(len(detectors))
        w = 0.35
        for ax, at in zip(axes, types):
            sub = rep_df[rep_df["attack_type"] == at]
            raw, res = [], []
            for d in detectors:
                r = sub[(sub["detector"] == d) & (sub["representation"] == "raw")][metric]
                s = sub[(sub["detector"] == d) & (sub["representation"] == "residual")][metric]
                raw.append(float(r.mean()) if len(r) else 0.0)
                res.append(float(s.mean()) if len(s) else 0.0)
            ax.bar(x - w / 2, raw, w, label="raw", color="steelblue", alpha=0.85)
            ax.bar(x + w / 2, res, w, label="residual", color="darkorange", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(detectors, rotation=45, ha="right", fontsize=6)
            ax.set_title(at.replace("_", "\n"), fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(metric)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Phase B ROAD representation: {metric}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"road_representation_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_phase_b(config: dict) -> dict:
    seed = int(config.get("seed", 0))
    set_global_seed(seed)
    tables_dir = Path(config.get("tables_dir", "tables"))
    figures_dir = Path(config.get("figures_dir", "figures"))
    exp_root = Path(config.get("output_dir", "experiments"))
    run = ExperimentRun(exp_root, name="phase_b_road")
    setup_logging(run.path("phase_b_road.log"))
    run.save_config(config)

    road_root = resolve_road_root(config["data_dir"])
    logger.info("ROAD root: %s", road_root)
    cache = FeatureCache(config.get("cache_dir", ".cache/phase_b_road"))

    cs = config.get("capture_sample", {})
    captures = list_eval_captures(
        road_root,
        skip_masquerade=cs.get("skip_masquerade", True),
        skip_unlabeled=cs.get("skip_unlabeled", True),
        per_type=cs.get("per_type"),
    )
    logger.info("Evaluating %d captures", len(captures))

    detectors = config["detectors"]
    representations = config.get("representations", ["raw", "residual"])
    eval_cfg = config["evaluation"]
    boot_cfg = config.get("bootstrap", {})
    det_params = config.get("detector_params", {})
    max_frames = config.get("max_frames_per_capture")
    ws = int(config.get("window_size", 30))

    rows = []
    for cap_name, meta in captures:
        logger.info("---- prepare %s ----", cap_name)
        prep = prepare_road_capture(
            road_root,
            cap_name,
            meta,
            window_size=ws,
            max_frames=max_frames,
            cache=cache,
        )
        if prep.get("error"):
            logger.warning("skip %s: %s", cap_name, prep["error"])
            rows.append(
                {
                    "experiment": "road_representation",
                    "capture": cap_name,
                    "attack_type": prep.get("attack_type", cap_name),
                    "error": prep["error"],
                }
            )
            continue

        logger.info(
            "  pre=%d test=%d attack_test=%d ids=%d",
            prep["n_pre"],
            prep["n_test"],
            prep["n_test_attack"],
            prep["per_id_stats_n"],
        )

        for rep in representations:
            train_df = prep[rep]["train"]
            test_df = prep[rep]["test"]
            cols = prep["feature_cols"] if rep == "raw" else prep["residual_cols"]

            for det_name in detectors:
                logger.info("=== %s | %s | %s ===", cap_name, rep, det_name)
                params = dict(det_params.get(det_name, {}))
                if "random_state" in params:
                    params["random_state"] = seed
                try:
                    det = create_detector(det_name, **params)
                    out = train_anomaly_detector(
                        det,
                        train_df,
                        test_df,
                        cols,
                        val_holdout_fraction=eval_cfg.get("val_holdout_fraction", 0.2),
                        fpr_target=eval_cfg.get("fpr_target", 0.01),
                        measure_resources=True,
                    )
                    boot = bootstrap_metrics(
                        out["y_test"],
                        out["y_pred"],
                        out["scores_test"],
                        n_boot=int(boot_cfg.get("n_boot", 300)),
                        block_size=int(boot_cfg.get("block_size", 40)),
                        seed=int(boot_cfg.get("seed", seed)),
                        ci=float(boot_cfg.get("ci", 0.95)),
                    )
                    row = {
                        "experiment": "road_representation",
                        "capture": cap_name,
                        "attack_type": prep["attack_type"],
                        "representation": rep,
                        "detector": det_name,
                        "seed": seed,
                        "protocol": "pre_injection_v1",
                        "protocol_version": "pre_injection_v1",
                        "phase": "B",
                        "n_pre": prep["n_pre"],
                        "n_test": prep["n_test"],
                        "n_test_attack": prep["n_test_attack"],
                        "precision": out["precision"],
                        "recall": out["recall"],
                        "f1": out["f1"],
                        "roc_auc": out["roc_auc"],
                        "pr_auc": out["pr_auc"],
                        "fpr": out["fpr"],
                        "threshold": out["threshold"],
                        "n_train_normals": out["n_train_normals"],
                        "train_seconds": out.get("train_seconds"),
                        "score_seconds": out.get("score_seconds"),
                        "runtime_seconds": out.get("runtime_seconds"),
                        "peak_rss_mb": out.get("peak_rss_mb"),
                        "model_bytes": out.get("model_bytes"),
                        "tp": out["tp"],
                        "fp": out["fp"],
                        "fn": out["fn"],
                        "tn": out["tn"],
                    }
                    row.update(flatten_bootstrap(boot))
                    rows.append(row)
                    logger.info(
                        "  F1=%.3f Rec=%.3f FPR=%.4f ROC=%.3f",
                        out["f1"],
                        out["recall"],
                        out["fpr"],
                        out["roc_auc"],
                    )
                except Exception as exc:
                    logger.exception("FAILED %s/%s/%s", cap_name, rep, det_name)
                    rows.append(
                        {
                            "experiment": "road_representation",
                            "capture": cap_name,
                            "attack_type": prep["attack_type"],
                            "representation": rep,
                            "detector": det_name,
                            "seed": seed,
                            "error": str(exc),
                        }
                    )

    df = pd.DataFrame(rows)
    rep_df = df[df.get("representation").notna()] if "representation" in df.columns else df
    rep_df = rep_df[rep_df["f1"].notna()] if "f1" in rep_df.columns else rep_df

    # Deltas
    delta_rows = []
    if not rep_df.empty and "representation" in rep_df.columns:
        for (cap, det), g in rep_df.groupby(["capture", "detector"]):
            raw = g[g["representation"] == "raw"]
            res = g[g["representation"] == "residual"]
            if len(raw) != 1 or len(res) != 1:
                continue
            raw, res = raw.iloc[0], res.iloc[0]
            delta_rows.append(
                {
                    "capture": cap,
                    "attack_type": res.get("attack_type", cap),
                    "detector": det,
                    "delta_f1": res["f1"] - raw["f1"],
                    "delta_roc_auc": res["roc_auc"] - raw["roc_auc"],
                    "delta_pr_auc": res["pr_auc"] - raw["pr_auc"],
                    "delta_recall": res["recall"] - raw["recall"],
                    "raw_f1": raw["f1"],
                    "residual_f1": res["f1"],
                    "raw_roc_auc": raw["roc_auc"],
                    "residual_roc_auc": res["roc_auc"],
                    "residual_fpr": res["fpr"],
                    "raw_fpr": raw["fpr"],
                }
            )
    delta_df = pd.DataFrame(delta_rows)
    verdict = _hypothesis_verdict(rep_df) if not rep_df.empty else {"verdict": "incomplete"}

    tables_dir.mkdir(parents=True, exist_ok=True)
    save_csv(tables_dir / "road_representation_results.csv", rep_df)
    save_csv(tables_dir / "road_representation_delta.csv", delta_df)
    save_csv(tables_dir / "road_baseline_results.csv", rep_df[rep_df["representation"] == "residual"] if "representation" in rep_df.columns else rep_df)
    save_json(tables_dir / "road_hypothesis_verdict.json", verdict)

    save_csv(run.path("road_representation_results.csv"), rep_df)
    save_csv(run.path("road_representation_delta.csv"), delta_df)
    save_json(run.path("road_hypothesis_verdict.json"), verdict)

    (exp_root / "road_representation").mkdir(parents=True, exist_ok=True)
    save_csv(exp_root / "road_representation" / "results.csv", rep_df)
    save_csv(exp_root / "road_representation" / "delta.csv", delta_df)

    _plot_bars(rep_df, figures_dir / "representation")

    logger.info("ROAD hypothesis verdict: %s", verdict.get("verdict"))
    logger.info("Run dir: %s", run.run_dir)
    return {"run_dir": str(run.run_dir), "verdict": verdict, "n_rows": len(df)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase B ROAD residualization study")
    parser.add_argument("--config", default="experiments/configs/phase_b_road.yaml")
    args = parser.parse_args()
    result = run_phase_b(load_config(args.config))
    print("\n=== PHASE B ROAD COMPLETE ===")
    print("Verdict:", result["verdict"].get("verdict"))
    print("Run dir:", result["run_dir"])


if __name__ == "__main__":
    main()
