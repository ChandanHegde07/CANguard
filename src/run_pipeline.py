"""Run the full PIRD pipeline on HCRL and save figures + results to disk."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from canguard.data import get_loader
from canguard.detectors import IsolationForestDetector
from canguard.evaluation import (
    train_anomaly_detector,
    cross_attack_evaluate,
    sweep_thresholds,
)
from canguard.features import (
    BEHAVIORAL_FEATURES_V1 as FEATURES,
    FeaturePipeline,
    fit_known_ids_on_normal_prefix,
    fit_per_id_stats,
    temporal_split,
    transform_residuals,
)
from canguard.visualization import (
    plot_cross_attack_matrix,
    plot_roc_pr,
    plot_score_distribution_grid,
    plot_score_timeline,
    plot_threshold_sweep,
)

DATA_DIR = Path("HCRL Car-Hacking")
OUTROOT = Path("results")
FIGDIR = Path("figures")
SAMPLE_SIZE = 60000
WINDOW_SIZE = 30
NAMES = ["DoS", "Fuzzy", "RPM", "gear"]


def save(fig, name: str) -> None:
    path = FIGDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main() -> None:
    OUTROOT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print("== Load ==")
    samples = {
        name: get_loader("hcrl", DATA_DIR / f"{name}_dataset.csv").load(sample_size=SAMPLE_SIZE)
        for name in NAMES
    }
    for name, df in samples.items():
        print(f"  {name}: {len(df):,} rows  ({df['is_attack'].mean()*100:.1f}% attack)")

    print("\n== Build feature tables ==")
    feature_tables = {}
    for name, df in samples.items():
        known = fit_known_ids_on_normal_prefix(df)
        pipe = FeaturePipeline(window_size=WINDOW_SIZE, known_ids=known)
        feature_tables[name] = pipe.process_dataframe(df)
        pipe.reset()
        print(f"  {name}: {len(feature_tables[name]):,} windows")

    print("\n== Residual transform + splits ==")
    RES_COLS = [c + "_res" for c in FEATURES]
    raw_data, residual_data = {}, {}
    for name in NAMES:
        ft = feature_tables[name]
        calib, train, test = temporal_split(ft, 0.4, 0.2, 0.4)
        raw_data[name] = {"calib": calib, "train": train, "test": test}
        stats, gstats = fit_per_id_stats(calib, FEATURES)
        residual_data[name] = {
            "calib": transform_residuals(calib, stats, gstats, FEATURES),
            "train": transform_residuals(train, stats, gstats, FEATURES),
            "test": transform_residuals(test, stats, gstats, FEATURES),
            "stats": stats,
            "global_stats": gstats,
        }

    print("\n== Isolation Forest (primary) ==")
    results = {}
    for name in NAMES:
        rd = residual_data[name]
        det = IsolationForestDetector(n_estimators=200, random_state=0)
        results[name] = train_anomaly_detector(det, rd["train"], rd["test"], RES_COLS)
        r = results[name]
        print(f"  {name}: F1={r['f1']:.3f} Recall={r['recall']:.3f} "
              f"FPR={r['fpr']:.4f} ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f}")

    print("\n== Figures: score distributions ==")
    save(plot_score_distribution_grid(results, NAMES), "score_distributions.png")

    print("== Figures: ROC/PR per dataset ==")
    for name in NAMES:
        r = results[name]
        save(plot_roc_pr(r["y_test"], r["scores_test"], fpr_at_op=r["fpr"],
                         title_prefix=name + " "),
             f"roc_pr_{name.lower()}.png")

    print("== Figures: RPM timeline ==")
    r = results["RPM"]
    save(plot_score_timeline(r["scores_test"], r["y_test"], r["threshold"],
                             title="RPM test segment"),
         "timeline_rpm.png")

    print("\n== Cross-attack matrix ==")
    target_stats = {t: fit_per_id_stats(raw_data[t]["calib"], FEATURES) for t in NAMES}
    rows = []
    for s in NAMES:
        for t in NAMES:
            stats, gstats = target_stats[t]
            src_norm = raw_data[s]["train"]
            src_norm = src_norm[src_norm["is_attack"] == 0]
            src_res = transform_residuals(src_norm, stats, gstats, FEATURES)
            tgt_res = transform_residuals(raw_data[t]["test"], stats, gstats, FEATURES)
            det = IsolationForestDetector(n_estimators=200, random_state=0)
            out = cross_attack_evaluate(det, src_res, tgt_res, RES_COLS)
            rows.append({"src": s, "tgt": t, "recall": out["recall"], "fpr": out["fpr"]})
    cross_df = pd.DataFrame(rows)
    rec_mat = cross_df.pivot_table(index="src", columns="tgt", values="recall", aggfunc="first")
    fpr_mat = cross_df.pivot_table(index="src", columns="tgt", values="fpr", aggfunc="first")
    print(rec_mat.round(3).to_string())
    save(plot_cross_attack_matrix(rec_mat, fpr_mat), "cross_attack_matrix.png")

    print("\n== Threshold sweep (RPM, DoS) ==")
    sweeps = {}
    for name in ["RPM", "DoS"]:
        ft = feature_tables[name]
        calib, train, test = temporal_split(ft, 0.4, 0.2, 0.4)
        stats, gstats = fit_per_id_stats(calib, FEATURES)
        res_train = transform_residuals(train, stats, gstats, FEATURES)
        res_test = transform_residuals(test, stats, gstats, FEATURES)
        train_norm = res_train[res_train["is_attack"] == 0].copy()
        n_val = max(1, int(len(train_norm) * 0.2))
        det = IsolationForestDetector(n_estimators=200, random_state=0)
        det.fit(train_norm.iloc[:-n_val][RES_COLS].fillna(0).values)
        val_scores = det.score_samples(train_norm.iloc[-n_val:][RES_COLS].fillna(0).values)
        test_scores = det.score_samples(res_test[RES_COLS].fillna(0).values)
        y_test = res_test["is_attack"].values
        sweeps[name] = sweep_thresholds(val_scores, test_scores, y_test,
                                        [0.001, 0.01, 0.05])
        print(f"  {name} sweep:")
        print(pd.DataFrame(sweeps[name])[["target_FPR", "actual_FPR", "recall", "f1"]].round(4).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ["RPM", "DoS"]):
        plot_threshold_sweep(sweeps[name], title=name, ax=ax)
    fig.tight_layout()
    save(fig, "threshold_sweep.png")

    print("\n== Write results summary ==")
    summary = {
        "config": {
            "sample_size": SAMPLE_SIZE,
            "window_size": WINDOW_SIZE,
            "detector": "isolation_forest",
            "n_estimators": 200,
            "random_state": 0,
            "split": {"calib": 0.4, "train": 0.2, "test": 0.4},
            "fpr_target": 0.01,
        },
        "per_dataset": {
            name: {k: v for k, v in results[name].items()
                   if k in ("fpr", "precision", "recall", "f1", "roc_auc", "pr_auc",
                            "threshold", "n_train_normals")}
            for name in NAMES
        },
        "cross_attack": cross_df.to_dict(orient="records"),
        "threshold_sweeps": {name: sweeps[name] for name in sweeps},
    }
    out_path = OUTROOT / "pipeline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  wrote {out_path}")

    print("\nDONE")


if __name__ == "__main__":
    main()
