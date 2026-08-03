"""Re-run max_speedometer_attack_1 after frame-selection fix; merge into ROAD tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from canguard.detectors import create_detector
from canguard.evaluation import bootstrap_metrics, flatten_bootstrap, train_anomaly_detector
from canguard.exp.road_protocol import list_eval_captures, prepare_road_capture, resolve_road_root

DETECTORS = {
    "isolation_forest": dict(n_estimators=200, n_jobs=1, random_state=0),
    "one_class_svm": dict(nu=0.05, max_train_samples=4000, random_state=0),
    "lof": dict(n_neighbors=20, max_train_samples=6000, n_jobs=1, random_state=0),
    "hbos": dict(n_bins=10),
    "elliptic_envelope": dict(contamination=0.05, max_train_samples=6000, random_state=0),
    "autoencoder": dict(
        hidden_layer_sizes=(8, 4, 8), max_iter=80, max_train_samples=6000, random_state=0
    ),
}


def main() -> None:
    root = resolve_road_root("road/road")
    caps = dict(list_eval_captures(root, per_type=1))
    name = "max_speedometer_attack_1"
    meta = caps[name]
    prep = prepare_road_capture(root, name, meta, window_size=30, max_frames=80000)
    if prep.get("error"):
        raise SystemExit(f"prepare failed: {prep['error']}")
    print(f"pre={prep['n_pre']} test={prep['n_test']} attack={prep['n_test_attack']}")

    rows = []
    for rep in ("raw", "residual"):
        train = prep[rep]["train"]
        test = prep[rep]["test"]
        cols = prep["feature_cols"] if rep == "raw" else prep["residual_cols"]
        for det_name, params in DETECTORS.items():
            det = create_detector(det_name, **params)
            out = train_anomaly_detector(det, train, test, cols, measure_resources=True)
            boot = bootstrap_metrics(
                out["y_test"], out["y_pred"], out["scores_test"], n_boot=200, block_size=40, seed=0
            )
            row = {
                "experiment": "road_representation",
                "capture": name,
                "attack_type": "max_speedometer_attack",
                "representation": rep,
                "detector": det_name,
                "seed": 0,
                "protocol": "pre_injection_v1",
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
                "runtime_seconds": out.get("runtime_seconds"),
                "model_bytes": out.get("model_bytes"),
                "tp": out["tp"],
                "fp": out["fp"],
                "fn": out["fn"],
                "tn": out["tn"],
            }
            row.update(flatten_bootstrap(boot))
            rows.append(row)
            print(f"{rep:9s} {det_name:20s} F1={out['f1']:.3f} ROC={out['roc_auc']:.3f}")

    extra = pd.DataFrame(rows)
    tables = Path("tables")
    main_path = tables / "road_representation_results.csv"
    base = pd.read_csv(main_path)
    base = base[~base["capture"].eq(name)]
    merged = pd.concat([base, extra], ignore_index=True)
    merged.to_csv(main_path, index=False)

    # Recompute deltas + verdict
    delta_rows = []
    for (cap, det), g in merged.groupby(["capture", "detector"]):
        raw = g[g["representation"] == "raw"]
        res = g[g["representation"] == "residual"]
        if len(raw) != 1 or len(res) != 1:
            continue
        raw, res = raw.iloc[0], res.iloc[0]
        delta_rows.append(
            {
                "capture": cap,
                "attack_type": res["attack_type"],
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
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(tables / "road_representation_delta.csv", index=False)

    # Verdict
    pivot_f1 = merged.pivot_table(
        index=["capture", "detector"], columns="representation", values="f1", aggfunc="first"
    )
    pivot_roc = merged.pivot_table(
        index=["capture", "detector"], columns="representation", values="roc_auc", aggfunc="first"
    )
    d_f1 = (pivot_f1["residual"] - pivot_f1["raw"]).dropna()
    d_roc = (pivot_roc["residual"] - pivot_roc["raw"]).dropna()
    n = len(d_f1)
    frac_f1 = float((d_f1 > 0).mean())
    frac_roc = float((d_roc > 0).mean())
    median_df1 = float(d_f1.median())
    captures = d_f1.index.get_level_values("capture")
    is_fuzz = captures.str.contains("fuzzing")
    fuzz_frac = float((d_f1[is_fuzz] > 0).mean()) if is_fuzz.any() else float("nan")
    targ_frac = float((d_f1[~is_fuzz] > 0).mean()) if (~is_fuzz).any() else float("nan")
    if frac_f1 >= 0.70 and median_df1 > 0 and frac_roc >= 0.60:
        verdict_s = "A_supported"
    elif frac_f1 >= 0.50 or (median_df1 > 0 and targ_frac >= 0.75):
        verdict_s = "B_partially_supported"
    else:
        verdict_s = "C_rejected"
    verdict = {
        "verdict": verdict_s,
        "n_cells": n,
        "frac_f1_positive": frac_f1,
        "frac_roc_positive": frac_roc,
        "median_delta_f1": median_df1,
        "median_delta_roc_auc": float(d_roc.median()),
        "targeted_frac_f1_positive": targ_frac,
        "fuzzing_frac_f1_positive": fuzz_frac,
        "protocol": "pre_injection_v1",
        "note": "includes max_speedometer re-run after frame-selection fix",
    }
    (tables / "road_hypothesis_verdict.json").write_text(json.dumps(verdict, indent=2))
    residual = merged[merged["representation"] == "residual"]
    residual.to_csv(tables / "road_baseline_results.csv", index=False)
    print("Verdict:", verdict_s, "n_cells=", n, "frac_f1=", round(frac_f1, 3))
    print("Updated tables/road_*.csv")


if __name__ == "__main__":
    main()
