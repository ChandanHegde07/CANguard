"""Regenerate key workshop figures with shared IEEE style (presentation only).

Usage:
    python experiments/scripts/regenerate_ieee_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from canguard.visualization.style import (
    IEEE_COLORS,
    apply_ieee_style,
    figsize_double,
    figsize_single,
    save_ieee_figure,
)


def main() -> None:
    apply_ieee_style()
    tables = _ROOT / "tables"
    fig_root = _ROOT / "figures"
    paper_fig = _ROOT / "paper" / "updated_figures"
    for d in [
        fig_root / "representation",
        fig_root / "feature_importance",
        fig_root / "runtime",
        fig_root / "latency",
        paper_fig,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Representation multi-seed delta F1
    ci_path = tables / "confidence_intervals.csv"
    if ci_path.exists():
        ci = pd.read_csv(ci_path)
        for corpus in ci["corpus"].dropna().unique():
            sub = ci[ci["corpus"] == corpus]
            if "f1_mean" not in sub.columns:
                continue
            piv = sub.pivot_table(
                index="detector", columns="representation", values="f1_mean", aggfunc="mean"
            )
            if "raw" not in piv.columns or "residual" not in piv.columns:
                continue
            delta = (piv["residual"] - piv["raw"]).sort_values()
            fig, ax = plt.subplots(figsize=figsize_single(2.6))
            colors = [
                IEEE_COLORS["residual"] if v >= 0 else IEEE_COLORS["attack"] for v in delta.values
            ]
            ax.barh(delta.index.astype(str), delta.values, color=colors, height=0.7)
            ax.axvline(0, color=IEEE_COLORS["neutral"], lw=0.7)
            ax.set_xlabel(r"Mean $\Delta$F1 (residual $-$ raw)")
            ax.set_title(f"Representation gain ({corpus.upper()})")
            for dest in (fig_root / "representation", paper_fig):
                save_ieee_figure(fig, dest / f"multiseed_delta_f1_{corpus}.png", close=False)
            plt.close(fig)

    # Feature importance
    imp_path = tables / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        for (corpus, dataset), g in imp.groupby(["corpus", "dataset"]):
            g = g.sort_values("importance_mean").tail(10)
            fig, ax = plt.subplots(figsize=figsize_single(2.8))
            ax.barh(
                g["feature"].astype(str),
                g["importance_mean"],
                xerr=g.get("importance_std"),
                color=IEEE_COLORS["residual"],
                height=0.7,
            )
            ax.set_xlabel(r"Permutation importance ($\Delta$ROC-AUC)")
            ax.set_title(f"{corpus}: {dataset}")
            safe = str(dataset).replace("/", "_")[:40]
            for dest in (fig_root / "feature_importance", paper_fig):
                save_ieee_figure(fig, dest / f"perm_importance_{corpus}_{safe}.png", close=False)
            plt.close(fig)

    # Runtime
    rt_path = tables / "runtime.csv"
    if rt_path.exists():
        rt = pd.read_csv(rt_path)
        sub = rt[rt["representation"] == "residual"] if "representation" in rt.columns else rt
        if len(sub) and "detector" in sub.columns:
            fig, ax = plt.subplots(figsize=figsize_single(2.5))
            pivot = sub.groupby("detector")[["train_seconds", "score_seconds"]].mean()
            x = np.arange(len(pivot))
            w = 0.35
            ax.bar(x - w / 2, pivot["train_seconds"], w, label="train", color=IEEE_COLORS["raw"])
            ax.bar(
                x + w / 2, pivot["score_seconds"], w, label="score", color=IEEE_COLORS["residual"]
            )
            ax.set_xticks(x)
            ax.set_xticklabels(pivot.index, rotation=30, ha="right")
            ax.set_ylabel("Seconds")
            ax.set_title("Train / score time (residual)")
            ax.legend()
            for dest in (fig_root / "runtime", paper_fig):
                save_ieee_figure(fig, dest / "runtime_train_score.png", close=False)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=figsize_single(2.5))
            wp = sub.groupby("detector")["windows_per_sec"].mean().sort_values()
            ax.barh(wp.index.astype(str), wp.values, color=IEEE_COLORS["raw"], height=0.7)
            ax.set_xlabel("Windows / s (scoring)")
            ax.set_title("Inference throughput")
            for dest in (fig_root / "runtime", paper_fig):
                save_ieee_figure(fig, dest / "runtime_throughput.png", close=False)
            plt.close(fig)

    # Latency
    lat_path = tables / "latency.csv"
    if lat_path.exists():
        lat = pd.read_csv(lat_path)
        if "delay_ms" in lat.columns:
            vals = lat.loc[lat.get("detected", True) == True, "delay_ms"].dropna()
            if len(vals):
                fig, ax = plt.subplots(figsize=figsize_single(2.4))
                ax.hist(vals, bins=25, color=IEEE_COLORS["attack"], alpha=0.9, edgecolor="white")
                ax.set_xlabel("Detection latency (ms)")
                ax.set_ylabel("Count")
                ax.set_title("Detection latency (residual)")
                for dest in (fig_root / "latency", paper_fig):
                    save_ieee_figure(fig, dest / "latency_hist.png", close=False)
                plt.close(fig)

    # Coverage boundary if present
    rob = tables / "robustness_summary.csv"
    if rob.exists():
        r = pd.read_csv(rob)
        if "coverage_bucket" in r.columns and "f1_mean" in r.columns:
            fig, ax = plt.subplots(figsize=figsize_double(2.6))
            corpora = sorted(r["corpus"].unique())
            buckets = list(r["coverage_bucket"].unique())
            x = np.arange(len(buckets))
            w = 0.35
            for i, corpus in enumerate(corpora):
                g = r[r["corpus"] == corpus].set_index("coverage_bucket").reindex(buckets)
                ax.bar(
                    x + (i - 0.5) * w,
                    g["f1_mean"].fillna(0),
                    w,
                    yerr=g["f1_std"].fillna(0) if "f1_std" in g else None,
                    label=corpus,
                    color=IEEE_COLORS["raw"] if corpus == "hcrl" else IEEE_COLORS["residual"],
                )
            ax.set_xticks(x)
            ax.set_xticklabels([b.replace("_", "\n") for b in buckets], fontsize=6)
            ax.set_ylabel("Mean residual F1")
            ax.set_title("Coverage boundary by attack family")
            ax.legend()
            for dest in (fig_root / "representation", paper_fig):
                save_ieee_figure(fig, dest / "coverage_boundary.png", close=False)
            plt.close(fig)

    print("IEEE figures regenerated under figures/ and paper/updated_figures/")


if __name__ == "__main__":
    main()
