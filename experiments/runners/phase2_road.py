"""Frozen ROAD external validation for the PIRD / global / hybrid pipeline.

This module does **not** change the frozen methodology. It only adapts the
*frozen* per-ID and global feature pipelines to the documented ROAD
per-capture, pre-injection calibration protocol:

    pre-injection normals  -> fit PIRD stats / global stats / detectors / thresholds
    post-injection traffic -> score (test)

No ROAD attack/test observation enters calibration. Global feature definitions,
``window_frames=200``, robust normalization, Isolation Forest parameters, the
Isolation Forest threshold protocol, the hybrid rules and the adaptive attack
parameters are reused verbatim from Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from canguard.data.road import RoadLoader  # noqa: E402
from canguard.evaluation import train_anomaly_detector  # noqa: E402
from canguard.exp import FeatureCache, save_csv, save_json  # noqa: E402
from canguard.exp.road_protocol import (  # noqa: E402
    build_road_window_table,
    list_eval_captures,
    load_capture_frames,
    resolve_road_root,
)
from canguard.features import (  # noqa: E402
    ALL_GLOBAL_FEATURES,
    GLOBAL_ABLATIONS,
    fit_per_id_stats,
    transform_residuals,
)
from canguard.hybrid import fit_score_normalizer  # noqa: E402
from canguard.visualization.hybrid import plot_detection_probability_multi  # noqa: E402
from canguard.visualization.style import (  # noqa: E402
    IEEE_COLORS,
    apply_ieee_style,
    save_ieee_figure,
)

from .phase2_hybrid import (  # noqa: E402
    _build_global_table,
    _fit_global_branch,
    _global_refs,
    _val_normal_scores,
    run_hybrid,
)
from .run_phase2 import _feature_cols, _make_detector  # noqa: E402


def _road_cache_key(name: str, root: Path, cfg: dict, n_frames: int) -> dict:
    log = root / "attacks" / f"{name}.log"
    return {
        "stage": "road_windows",
        "capture": name,
        "window_size": int(cfg.get("window_size", 30)),
        "max_frames": cfg.get("max_frames_per_capture"),
        "n_frames": int(n_frames),
        "mtime": log.stat().st_mtime if log.exists() else 0,
        # Frozen frame-selection protocol (same as Phase B).
        "protocol": "pre_injection_v2_inj_priority",
    }


def _prepare_road_hybrid_dataset(name: str, cfg: dict, cache: FeatureCache) -> dict:
    """Build the branch dict for one ROAD attack capture (pre-injection calib)."""
    feature_cols = _feature_cols(cfg)
    eval_cfg = cfg.get("evaluation", {})
    val_fraction = eval_cfg.get("val_holdout_fraction", 0.2)
    fpr_target = eval_cfg.get("fpr_target", 0.01)
    ws = int(cfg.get("window_size", 30))

    root = resolve_road_root(cfg["data_dir"])
    meta_all = RoadLoader(root)._meta_all
    meta = meta_all.get(name, {})
    interval = meta.get("injection_interval")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError(f"capture {name} lacks a valid injection_interval")

    df = load_capture_frames(
        root, name, meta, max_frames=cfg.get("max_frames_per_capture")
    )
    if df.empty:
        raise ValueError(f"empty capture {name}")
    ck = _road_cache_key(name, root, cfg, len(df))
    ft = build_road_window_table(df, window_size=ws, cache=cache, cache_key=ck)
    if ft.empty:
        raise ValueError(f"no windows for {name}")
    ft = ft.sort_values("timestamp").reset_index(drop=True)
    if "elapsed" not in ft.columns:
        ft["elapsed"] = ft["timestamp"] - ft["timestamp"].iloc[0]

    start = float(interval[0])
    pre = ft[ft["elapsed"] < start].reset_index(drop=True)
    test = ft[ft["elapsed"] >= start].reset_index(drop=True)
    min_pre = int(cfg.get("min_pre_windows", 50))
    if len(pre) < min_pre:
        raise ValueError(f"too few pre-injection windows for {name}: {len(pre)} < {min_pre}")

    # Ensure pre is a contiguous leading block (causal rolling across the
    # calibration/test boundary stays valid).
    wt = pd.concat([pre, test], ignore_index=True)
    test_start = len(pre)
    calib = pre.copy()
    train = pre.copy()
    calib_norm = calib[calib["is_attack"] == 0]

    # ---- PIRD branch (frozen) -------------------------------------------
    stats, gstats = fit_per_id_stats(calib, feature_cols)
    res_cols = [c + "_res" for c in feature_cols]
    res_pre = transform_residuals(pre, stats, gstats, feature_cols)
    res_test = transform_residuals(test, stats, gstats, feature_cols)
    out_p = train_anomaly_detector(
        _make_detector(cfg),
        res_pre,
        res_test,
        res_cols,
        val_holdout_fraction=val_fraction,
        fpr_target=fpr_target,
    )
    pird = {
        "model": out_p["model"],
        "threshold": float(out_p["threshold"]),
        "val": _val_normal_scores(out_p["model"], res_pre, res_cols, val_fraction),
        "metrics": out_p,
        "res_cols": res_cols,
        "stats": stats,
        "gstats": gstats,
    }
    pird["norm"] = fit_score_normalizer(pird["val"])

    # ---- Global branch (frozen 28 features) ------------------------------
    gcfg = cfg.get("global", {})
    refs = _global_refs(calib_norm)
    gft = _build_global_table(wt, refs, gcfg)
    gcalib = gft.iloc[:test_start].reset_index(drop=True)
    gtest = gft.iloc[test_start:].reset_index(drop=True)
    global_cols = list(gcfg.get("feature_cols") or ALL_GLOBAL_FEATURES)
    method = gcfg.get("normalization", "robust")
    glob = _fit_global_branch(
        gcalib, gcalib, gtest, global_cols, cfg, method, val_fraction, fpr_target
    )
    ablation: dict[str, dict] = {}
    if gcfg.get("run_ablation", True):
        for fam, cols in GLOBAL_ABLATIONS.items():
            ablation[fam] = _fit_global_branch(
                gcalib, gcalib, gtest, cols, cfg, method, val_fraction, fpr_target
            )

    return {
        "name": name,
        "feature_cols": feature_cols,
        "wt": wt,
        "calib": calib,
        "train": train,
        "test": test,
        "calib_norm": calib_norm,
        "refs": refs,
        "gft": gft,
        "gcalib": gcalib,
        "gtrain": gcalib,
        "gtest": gtest,
        "pird": pird,
        "global": glob,
        "ablation": ablation,
        "split": {"protocol": "pre_injection", "injection_start": start},
        "eval_cfg": eval_cfg,
        "gcfg": gcfg,
        "method": method,
        "val_fraction": val_fraction,
        "fpr_target": fpr_target,
        "test_start": test_start,
        "attack_type": _attack_type(name),
        "injection_id": meta.get("injection_id"),
    }


def _attack_type(name: str) -> str:
    parts = name.split("_")
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return name


def _road_attack_summary(conv: pd.DataFrame, capture_names: list[str]) -> pd.DataFrame:
    """Aggregate conventional metrics by ROAD attack type and branch."""
    if conv.empty:
        return pd.DataFrame()
    df = conv.copy()
    df["attack_type"] = df["dataset"].map(_attack_type)
    rows = []
    for (atype, rule), g in df.groupby(["attack_type", "rule"]):
        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        fn = int(g["fn"].sum())
        tn = int(g["tn"].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({
            "attack_type": atype, "rule": rule, "n_captures": int(g["dataset"].nunique()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "actual_fpr": fpr,
            "mean_roc_auc": float(g["roc_auc"].mean(skipna=True)),
            "mean_pr_auc": float(g["pr_auc"].mean(skipna=True)),
            "false_alarms_per_hour": float(g["false_alarms_per_hour"].mean(skipna=True)),
        })
    return pd.DataFrame(rows).sort_values(["attack_type", "rule"]).reset_index(drop=True)


def _plot_road_f1(summary: pd.DataFrame, path: Path) -> None:
    """Grouped F1 by ROAD attack type for PIRD / global / hybrid-max."""
    apply_ieee_style()
    types = sorted(summary["attack_type"].unique())
    rules = ["pird", "global", "max"]
    colors = {
        "pird": IEEE_COLORS["residual"],
        "global": IEEE_COLORS["raw"],
        "max": IEEE_COLORS["attack"],
    }
    x = np.arange(len(types))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * len(types)), 3.2))
    for i, rule in enumerate(rules):
        vals = [
            float(summary[(summary["attack_type"] == t) & (summary["rule"] == rule)]["f1"].mean())
            for t in types
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=rule, color=colors[rule])
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=6)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("ROAD external validation: F1 by attack type (1% target)")
    ax.legend(fontsize=7)
    save_ieee_figure(fig, path)


def run_road_external(cfg: dict, out_root: Path) -> dict:
    """Run the frozen global/hybrid pipeline on ROAD attack captures."""
    root = resolve_road_root(cfg["data_dir"])
    cs = cfg.get("capture_sample", {})
    captures = list_eval_captures(
        root,
        skip_masquerade=cs.get("skip_masquerade", True),
        skip_unlabeled=cs.get("skip_unlabeled", True),
        per_type=cs.get("per_type", 1),
    )
    capture_names = [n for n, _ in captures]
    if not capture_names:
        raise RuntimeError("No eligible ROAD captures found.")

    road_cfg = dict(cfg)
    road_cfg["datasets"] = capture_names
    road_cfg["dataset"] = "road"
    out_root.mkdir(parents=True, exist_ok=True)
    save_json(out_root / "config.json", road_cfg)

    result = run_hybrid(road_cfg, out_root, prepare=_prepare_road_hybrid_dataset)

    summary = _road_attack_summary(result["conventional"], capture_names)
    if len(summary):
        save_csv(out_root / "road_attack_summary.csv", summary)
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if len(summary):
        _plot_road_f1(summary, plot_dir / "road_external_f1.png")
    if len(result["adaptive"]):
        plot_detection_probability_multi(
            result["adaptive"],
            plot_dir / "road_external_adaptive_boundary.png",
            title="ROAD: adaptive detection boundary (PIRD vs global vs hybrid)",
        )
        plot_detection_probability_multi(
            result["adaptive"],
            plot_dir / "road_external_adaptive_boundary_significant.png",
            significant=True,
            title="ROAD: adaptive boundary (control-adjusted)",
        )
    save_json(
        out_root / "road_external_summary.json",
        {
            "dataset": "road",
            "protocol": "pre_injection",
            "captures": capture_names,
            "attack_types": sorted({_attack_type(n) for n in capture_names}),
            "n_captures": len(capture_names),
            "n_adaptive_rows": int(len(result["adaptive"])),
        },
    )
    return result


__all__ = ["run_road_external", "_prepare_road_hybrid_dataset"]
