"""Publication-quality figures for the Phase 2 global/hybrid experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .style import IEEE_COLORS, apply_ieee_style, figsize_double, figsize_single, save_ieee_figure

RULE_COLORS = {
    "pird": IEEE_COLORS["residual"],
    "global": IEEE_COLORS["raw"],
    "or": IEEE_COLORS["secondary"],
    "max": IEEE_COLORS["attack"],
    "weighted_0.5": IEEE_COLORS["normal"],
}


def plot_detection_probability_multi(
    boundary_df: pd.DataFrame, path: str | Path, *, significant: bool = False,
    title: str = "Adaptive attack detection boundary",
) -> Path:
    apply_ieee_style()
    ycol = "detection_probability_significant" if significant else "detection_probability"
    fig, ax = plt.subplots(figsize=figsize_single(2.7))
    for rule, g in boundary_df.groupby("rule"):
        if rule.startswith("global_"):
            continue
        g = g.sort_values("severity")
        ax.plot(g["severity"], g[ycol], marker="o", color=RULE_COLORS.get(rule, None), label=rule)
    ax.axhline(0.5, color=IEEE_COLORS["neutral"], ls="--", lw=0.8, alpha=0.7)
    ax.axhline(0.9, color=IEEE_COLORS["neutral"], ls=":", lw=0.8, alpha=0.7)
    ax.set_xlabel(r"Attack severity $\alpha$")
    ax.set_ylabel("Detection probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=6)
    return save_ieee_figure(fig, path)


def plot_roc_pr_scores(
    scores_df: pd.DataFrame, path: str | Path, *, dataset: str,
    local_norm: dict | None = None, global_norm: dict | None = None,
) -> Path:
    apply_ieee_style()
    y = scores_df["y"].to_numpy(dtype=int)
    pird = scores_df["pird_score"].to_numpy(dtype=float)
    glob = scores_df["global_score"].to_numpy(dtype=float)
    curves = {"pird": pird, "global": glob}
    if local_norm and global_norm:
        zl = (pird - local_norm["center"]) / (local_norm["scale"] + 1e-9)
        zg = (glob - global_norm["center"]) / (global_norm["scale"] + 1e-9)
        curves["hybrid_max"] = np.maximum(zl, zg)
    fig, axes = plt.subplots(1, 2, figsize=figsize_double(2.8))
    for name, s in curves.items():
        if len(np.unique(y)) < 2:
            continue
        color = RULE_COLORS.get(name.replace("hybrid_", ""), IEEE_COLORS["neutral"])
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, color=color,
                     label=f"{name} (AUC={np.trapezoid(tpr, fpr):.3f})")
        prec, rec, _ = precision_recall_curve(y, s)
        axes[1].plot(rec, prec, color=color, label=name)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"{dataset}: ROC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{dataset}: PR")
    for ax in axes:
        ax.legend(fontsize=6)
    return save_ieee_figure(fig, path)


def plot_complementarity(comp_df: pd.DataFrame, path: str | Path) -> Path:
    apply_ieee_style()
    df = comp_df[comp_df["subset"] == "all"].copy()
    if df.empty:
        df = comp_df.copy()
    fig, ax = plt.subplots(figsize=figsize_single(2.7))
    datasets = list(df["dataset"])
    x = np.arange(len(datasets))
    cats = ["both", "pird_only", "global_only", "neither"]
    colors = [
        IEEE_COLORS["normal"],
        IEEE_COLORS["residual"],
        IEEE_COLORS["raw"],
        IEEE_COLORS["neutral"],
    ]
    bottom = np.zeros(len(datasets))
    for cat, color in zip(cats, colors):
        vals = df[cat].to_numpy(dtype=float)
        tot = df[["both", "pird_only", "global_only", "neither"]].sum(axis=1).to_numpy(dtype=float)
        frac = np.divide(vals, tot, out=np.zeros_like(vals), where=tot > 0)
        ax.bar(x, frac, bottom=bottom, color=color, label=cat)
        bottom += frac
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Fraction of test windows")
    ax.set_title("PIRD vs global detection overlap")
    ax.legend(fontsize=6, ncol=2)
    return save_ieee_figure(fig, path)


def plot_gradual_scores(traj_df: pd.DataFrame, path: str | Path, *, title: str) -> Path:
    apply_ieee_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_double(3.8), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    t = traj_df["t"].to_numpy(dtype=float)
    ax1.plot(t, traj_df["pird_score"], color=RULE_COLORS["pird"], label="PIRD score")
    ax1.plot(t, traj_df["global_score"], color=RULE_COLORS["global"], ls="--", label="Global score")
    ax1.set_ylabel("Raw anomaly score")
    ax1.legend(fontsize=6)
    ax2.plot(t, traj_df["hybrid_max_z"], color=RULE_COLORS["max"], label="Hybrid (max, z)")
    ax2.set_ylabel("Normalized hybrid score")
    ax2.set_xlabel("Window index within drift horizon")
    ax2.legend(fontsize=6)
    for ax in (ax1, ax2):
        ax.axvline(0, color=IEEE_COLORS["neutral"], ls=":", lw=1.0)
    ax1.set_title(title)
    return save_ieee_figure(fig, path)


def plot_fpr_recall(curve_df: pd.DataFrame, path: str | Path) -> Path:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=figsize_single(2.7))
    for rule, g in curve_df.groupby("rule"):
        if rule.startswith("global_"):
            continue
        agg = g.groupby("actual_fpr")["recall"].mean().sort_index()
        ax.plot(agg.index.to_numpy(dtype=float), agg.to_numpy(dtype=float), marker="o",
                ms=3, color=RULE_COLORS.get(rule, None), label=rule)
    ax.set_xscale("log")
    ax.set_xlabel("Actual FPR")
    ax.set_ylabel("Recall (mean across datasets)")
    ax.set_title("Recall vs false-positive rate")
    ax.legend(fontsize=6)
    return save_ieee_figure(fig, path)


def make_hybrid_plots(cfg: dict, out_root: Path, result: dict) -> None:
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    adaptive = result.get("adaptive")
    if adaptive is not None and len(adaptive):
        plot_detection_probability_multi(
            adaptive, plot_dir / "hybrid_detection_probability_vs_severity.png",
            title="Adaptive evasion: PIRD vs global vs hybrid",
        )
        plot_detection_probability_multi(
            adaptive, plot_dir / "hybrid_detection_probability_significant.png",
            significant=True, title="Adaptive evasion (control-adjusted)",
        )
    # ROC/PR per dataset
    norms_path = out_root / "hybrid" / "score_norms.json"
    norms = {}
    if norms_path.exists():
        import json

        norms = json.loads(norms_path.read_text())
    for ds, nrm in norms.items():
        sp = out_root / "hybrid" / f"scores_{ds}.csv"
        if sp.exists():
            plot_roc_pr_scores(
                pd.read_csv(sp), plot_dir / f"hybrid_roc_pr_{ds}.png", dataset=ds,
                local_norm={"center": nrm["local_center"], "scale": nrm["local_scale"]},
                global_norm={"center": nrm["global_center"], "scale": nrm["global_scale"]},
            )
    comp = result.get("complementarity")
    if comp is not None and len(comp):
        plot_complementarity(comp, plot_dir / "hybrid_complementarity.png")
    traj = result.get("trajectories")
    if traj is not None and len(traj):
        rep = traj[(traj["rate"] == "slow") & (traj["alpha_end"] == traj["alpha_end"].max())]
        if len(rep):
            ds0 = rep["dataset"].iloc[0]
            rep = rep[(rep["dataset"] == ds0) & (rep["target_id"] == rep["target_id"].iloc[0])
                      & (rep["direction"] == rep["direction"].iloc[0])]
            plot_gradual_scores(rep, plot_dir / "hybrid_gradual_scores.png",
                                title=f"Gradual drift ({ds0})")
    curve_path = out_root / "hybrid" / "fpr_recall_curve.csv"
    if curve_path.exists():
        plot_fpr_recall(pd.read_csv(curve_path), plot_dir / "hybrid_fpr_recall.png")


__all__ = [
    "make_hybrid_plots",
    "plot_complementarity",
    "plot_detection_probability_multi",
    "plot_fpr_recall",
    "plot_gradual_scores",
    "plot_roc_pr_scores",
]
