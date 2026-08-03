"""Phase A runner: Exp 1 baselines, Exp 2 representation, Exp 7 CIs, Exp 18 verdict.

Usage:
    python -m experiments.runners.run_phase_a --config experiments/configs/phase_a.yaml
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

# Ensure src/ is importable when run as module from repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.detectors import create_detector
from canguard.evaluation import bootstrap_metrics, flatten_bootstrap, train_anomaly_detector
from canguard.exp import (
    ExperimentRun,
    FeatureCache,
    load_config,
    set_global_seed,
    setup_logging,
    save_csv,
    save_json,
)
from canguard.exp.matrix import build_raw_and_residual_splits, build_window_table, resolve_data_path
from canguard.exp.metadata import PHASE_A, PROTOCOL_HCRL_TEMPORAL, tag_dataframe
from canguard.visualization import plot_roc_pr

logger = logging.getLogger("canguard")

METRIC_KEYS = ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr")


def _row_from_result(
    *,
    experiment: str,
    dataset: str,
    representation: str,
    detector: str,
    seed: int,
    out: dict,
    boot: dict | None,
) -> dict:
    row = {
        "experiment": experiment,
        "dataset": dataset,
        "representation": representation,
        "detector": detector,
        "seed": seed,
        "precision": out["precision"],
        "recall": out["recall"],
        "f1": out["f1"],
        "roc_auc": out["roc_auc"],
        "pr_auc": out["pr_auc"],
        "fpr": out["fpr"],
        "threshold": out["threshold"],
        "n_train_normals": out["n_train_normals"],
        "train_seconds": out.get("train_seconds", float("nan")),
        "score_seconds": out.get("score_seconds", float("nan")),
        "runtime_seconds": out.get("runtime_seconds", float("nan")),
        "peak_rss_mb": out.get("peak_rss_mb", float("nan")),
        "model_bytes": out.get("model_bytes", 0),
        "tp": out["tp"],
        "fp": out["fp"],
        "fn": out["fn"],
        "tn": out["tn"],
    }
    if boot is not None:
        row.update(flatten_bootstrap(boot))
    return row


def _hypothesis_verdict(rep_df: pd.DataFrame) -> dict:
    """Exp 18: does residualization improve across detectors?

    Uses paired residual - raw deltas on F1 and ROC-AUC per (dataset, detector).
    """
    pivot_f1 = rep_df.pivot_table(
        index=["dataset", "detector"], columns="representation", values="f1", aggfunc="first"
    )
    pivot_roc = rep_df.pivot_table(
        index=["dataset", "detector"], columns="representation", values="roc_auc", aggfunc="first"
    )
    if "raw" not in pivot_f1.columns or "residual" not in pivot_f1.columns:
        return {"verdict": "incomplete", "reason": "missing raw or residual rows"}

    d_f1 = (pivot_f1["residual"] - pivot_f1["raw"]).dropna()
    d_roc = (pivot_roc["residual"] - pivot_roc["raw"]).dropna()

    n = len(d_f1)
    n_f1_pos = int((d_f1 > 0).sum())
    n_roc_pos = int((d_roc > 0).sum())
    frac_f1 = n_f1_pos / n if n else 0.0
    frac_roc = n_roc_pos / n if n else 0.0
    median_df1 = float(d_f1.median()) if n else float("nan")
    median_droc = float(d_roc.median()) if n else float("nan")

    # Pre-registered decision rule (EXPERIMENT_PLAN / Phase A):
    # Supported: residual better on F1 for >= 70% of cells AND median ΔF1 > 0
    #            AND residual better on ROC for >= 60% of cells
    # Partially: residual better on F1 for >= 50% OR median ΔF1 > 0 with clear spoof wins
    # Rejected: otherwise
    spoof = d_f1.loc[d_f1.index.get_level_values("dataset").isin(["RPM", "gear"])]
    spoof_frac = float((spoof > 0).mean()) if len(spoof) else 0.0

    if frac_f1 >= 0.70 and median_df1 > 0 and frac_roc >= 0.60:
        verdict = "A_supported"
    elif frac_f1 >= 0.50 or (median_df1 > 0 and spoof_frac >= 0.75):
        verdict = "B_partially_supported"
    else:
        verdict = "C_rejected"

    per_cell = []
    for (ds, det), df1 in d_f1.items():
        per_cell.append(
            {
                "dataset": ds,
                "detector": det,
                "delta_f1": float(df1),
                "delta_roc_auc": float(d_roc.loc[(ds, det)]) if (ds, det) in d_roc.index else float("nan"),
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
        "spoof_frac_f1_positive": spoof_frac,
        "n_f1_positive": n_f1_pos,
        "n_roc_positive": n_roc_pos,
        "per_cell": per_cell,
        "decision_rule": {
            "A_supported": "frac_f1>=0.70 & median_dF1>0 & frac_roc>=0.60",
            "B_partially_supported": "frac_f1>=0.50 OR (median_dF1>0 & spoof_frac>=0.75)",
            "C_rejected": "otherwise",
        },
    }


def _plot_representation_bars(rep_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("f1", "roc_auc", "pr_auc"):
        datasets = sorted(rep_df["dataset"].unique())
        detectors = sorted(rep_df["detector"].unique())
        fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
        if len(datasets) == 1:
            axes = [axes]
        x = np.arange(len(detectors))
        width = 0.35
        for ax, ds in zip(axes, datasets):
            sub = rep_df[rep_df["dataset"] == ds]
            raw = [
                float(sub[(sub["detector"] == d) & (sub["representation"] == "raw")][metric].values[0])
                if len(sub[(sub["detector"] == d) & (sub["representation"] == "raw")])
                else 0.0
                for d in detectors
            ]
            res = [
                float(
                    sub[(sub["detector"] == d) & (sub["representation"] == "residual")][metric].values[0]
                )
                if len(sub[(sub["detector"] == d) & (sub["representation"] == "residual")])
                else 0.0
                for d in detectors
            ]
            ax.bar(x - width / 2, raw, width, label="raw", color="steelblue", alpha=0.85)
            ax.bar(x + width / 2, res, width, label="residual", color="darkorange", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(detectors, rotation=45, ha="right", fontsize=7)
            ax.set_title(ds)
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", alpha=0.3)
        axes[0].set_ylabel(metric)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Phase A representation study: {metric}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"representation_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_phase_a(config: dict) -> dict:
    seed = int(config.get("seed", 0))
    set_global_seed(seed)
    tables_dir = Path(config.get("tables_dir", "tables"))
    figures_dir = Path(config.get("figures_dir", "figures"))
    exp_root = Path(config.get("output_dir", "experiments"))
    run = ExperimentRun(exp_root, name="phase_a")
    setup_logging(run.path("phase_a.log"))
    run.save_config(config)
    cache = FeatureCache(config.get("cache_dir", ".cache/phase_a"))

    datasets = config["datasets"]
    detectors = config["detectors"]
    representations = config.get("representations", ["raw", "residual"])
    split = config["split"]
    eval_cfg = config["evaluation"]
    boot_cfg = config.get("bootstrap", {})
    det_params = config.get("detector_params", {})

    # Pre-build matrices once per dataset
    matrices = {}
    for name in datasets:
        csv_path = resolve_data_path(config["data_dir"], name)
        ft = build_window_table(
            csv_path,
            window_size=int(config.get("window_size", 30)),
            sample_size=config.get("sample_size"),
            cache=cache,
        )
        matrices[name] = build_raw_and_residual_splits(
            ft,
            calib_frac=split["calib_frac"],
            train_frac=split["train_frac"],
            test_frac=split["test_frac"],
        )
        logger.info(
            "[%s] raw train=%d test=%d residual IDs fitted=%d",
            name,
            len(matrices[name]["raw"]["train"]),
            len(matrices[name]["raw"]["test"]),
            matrices[name]["per_id_stats_n"],
        )

    all_rows = []
    score_store = {}  # for ROC/PR figures

    for name in datasets:
        mat = matrices[name]
        for rep in representations:
            if rep == "raw":
                train_df = mat["raw"]["train"]
                test_df = mat["raw"]["test"]
                cols = mat["feature_cols"]
            else:
                train_df = mat["residual"]["train"]
                test_df = mat["residual"]["test"]
                cols = mat["residual_cols"]

            for det_name in detectors:
                logger.info("=== %s | %s | %s ===", name, rep, det_name)
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
                        n_boot=int(boot_cfg.get("n_boot", 400)),
                        block_size=int(boot_cfg.get("block_size", 50)),
                        seed=int(boot_cfg.get("seed", seed)),
                        ci=float(boot_cfg.get("ci", 0.95)),
                    )
                    experiment = "baseline" if rep == "residual" else "representation"
                    # Both reps go into representation study; residual-only also baseline.
                    row = _row_from_result(
                        experiment="representation",
                        dataset=name,
                        representation=rep,
                        detector=det_name,
                        seed=seed,
                        out=out,
                        boot=boot,
                    )
                    all_rows.append(row)
                    if rep == "residual":
                        base_row = dict(row)
                        base_row["experiment"] = "baseline"
                        all_rows.append(base_row)

                    score_store[(name, rep, det_name)] = {
                        "y_test": out["y_test"],
                        "scores": out["scores_test"],
                        "fpr": out["fpr"],
                    }
                    logger.info(
                        "  F1=%.3f Rec=%.3f FPR=%.4f ROC=%.3f PR=%.3f runtime=%.2fs",
                        out["f1"],
                        out["recall"],
                        out["fpr"],
                        out["roc_auc"],
                        out["pr_auc"],
                        out.get("runtime_seconds", 0),
                    )
                except Exception as exc:
                    logger.exception("FAILED %s/%s/%s: %s", name, rep, det_name, exc)
                    all_rows.append(
                        {
                            "experiment": "representation",
                            "dataset": name,
                            "representation": rep,
                            "detector": det_name,
                            "seed": seed,
                            "precision": float("nan"),
                            "recall": float("nan"),
                            "f1": float("nan"),
                            "roc_auc": float("nan"),
                            "pr_auc": float("nan"),
                            "fpr": float("nan"),
                            "error": str(exc),
                        }
                    )

    df = pd.DataFrame(all_rows)
    df = tag_dataframe(df, protocol_version=PROTOCOL_HCRL_TEMPORAL, phase=PHASE_A)
    rep_df = df[df["experiment"] == "representation"].copy()
    base_df = df[df["experiment"] == "baseline"].copy()

    # Deltas for representation study
    delta_rows = []
    for (ds, det), g in rep_df.groupby(["dataset", "detector"]):
        raw = g[g["representation"] == "raw"]
        res = g[g["representation"] == "residual"]
        if len(raw) != 1 or len(res) != 1:
            continue
        raw, res = raw.iloc[0], res.iloc[0]
        delta_rows.append(
            {
                "dataset": ds,
                "detector": det,
                "delta_f1": res["f1"] - raw["f1"],
                "delta_roc_auc": res["roc_auc"] - raw["roc_auc"],
                "delta_pr_auc": res["pr_auc"] - raw["pr_auc"],
                "delta_precision": res["precision"] - raw["precision"],
                "delta_recall": res["recall"] - raw["recall"],
                "raw_f1": raw["f1"],
                "residual_f1": res["f1"],
                "raw_roc_auc": raw["roc_auc"],
                "residual_roc_auc": res["roc_auc"],
            }
        )
    delta_df = pd.DataFrame(delta_rows)

    verdict = _hypothesis_verdict(rep_df)

    # Save tables (run dir + canonical tables/)
    tables_dir.mkdir(parents=True, exist_ok=True)
    save_csv(tables_dir / "baseline_results.csv", base_df)
    save_csv(tables_dir / "representation_results.csv", rep_df)
    save_csv(tables_dir / "representation_delta.csv", delta_df)
    # Statistics: long form CIs from representation rows
    stat_cols = [c for c in rep_df.columns if any(c.startswith(m + "_") for m in METRIC_KEYS)]
    stats_df = rep_df[
        ["dataset", "representation", "detector", "seed"]
        + [c for c in ["precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"] if c in rep_df.columns]
        + stat_cols
    ].copy()
    save_csv(tables_dir / "statistics.csv", stats_df)

    save_csv(run.path("baseline_results.csv"), base_df)
    save_csv(run.path("representation_results.csv"), rep_df)
    save_csv(run.path("representation_delta.csv"), delta_df)
    save_csv(run.path("statistics.csv"), stats_df)
    save_json(run.path("hypothesis_verdict.json"), verdict)
    save_json(tables_dir / "hypothesis_verdict.json", verdict)

    # Experiment folder copies
    for sub, frame in [
        ("baselines", base_df),
        ("representation", rep_df),
        ("statistics", stats_df),
    ]:
        d = exp_root / sub
        d.mkdir(parents=True, exist_ok=True)
        save_csv(d / "results.csv", frame)

    # Figures
    roc_dir = figures_dir / "roc"
    pr_dir = figures_dir / "pr"
    rep_fig = figures_dir / "representation"
    roc_dir.mkdir(parents=True, exist_ok=True)
    pr_dir.mkdir(parents=True, exist_ok=True)

    for (name, rep, det_name), store in score_store.items():
        try:
            fig = plot_roc_pr(
                store["y_test"],
                store["scores"],
                fpr_at_op=store["fpr"],
                title_prefix=f"{name} {rep} {det_name} ",
            )
            fig.savefig(
                roc_dir / f"{name}_{rep}_{det_name}_roc_pr.png",
                dpi=120,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as exc:
            logger.warning("figure failed %s: %s", (name, rep, det_name), exc)

    _plot_representation_bars(rep_df, rep_fig)

    logger.info("Hypothesis verdict: %s", verdict["verdict"])
    logger.info("Phase A run dir: %s", run.run_dir)
    return {"run_dir": str(run.run_dir), "verdict": verdict, "n_rows": len(df)}


def main() -> None:
    parser = argparse.ArgumentParser(description="CANguard Phase A experiments")
    parser.add_argument(
        "--config",
        default="experiments/configs/phase_a.yaml",
        help="Path to phase_a YAML config",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_phase_a(cfg)
    print("\n=== PHASE A COMPLETE ===")
    print("Verdict:", result["verdict"]["verdict"])
    print("Run dir:", result["run_dir"])


if __name__ == "__main__":
    main()
