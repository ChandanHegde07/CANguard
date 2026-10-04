"""Phase 3 transfer visualisation: heatmaps, ID overlap, SMD."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import IEEE_COLORS, apply_ieee_style, figsize_double, figsize_single, save_ieee_figure


def plot_transfer_heatmap(
    df: pd.DataFrame, metric: str, path: str | Path, *, title: str | None = None,
    cmap: str = "viridis", vmin: float = 0.0, vmax: float = 1.0,
) -> Path:
    """Source (rows) × target (columns) heatmap of ``metric``."""
    apply_ieee_style()
    if df.empty or metric not in df.columns:
        fig, ax = plt.subplots(figsize=figsize_single(2.2))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return save_ieee_figure(fig, path)
    piv = df.pivot_table(index="source_dataset", columns="target_dataset",
                         values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=figsize_single(2.4))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if vmax and v > 0.5 * vmax else "black")
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_title(title or metric)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_ieee_figure(fig, path)


def plot_id_overlap(df: pd.DataFrame, name_col: str, path: str | Path, *, title: str) -> Path:
    """Stacked shared / source-only / target-only ID counts per transfer."""
    apply_ieee_style()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize_single(2.2))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return save_ieee_figure(fig, path)
    labels = df[name_col].astype(str).tolist()
    shared = df["n_shared_ids"].to_numpy(dtype=float)
    so = df["n_source_only_ids"].to_numpy(dtype=float)
    to = df["n_target_only_ids"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize_single(3.2))
    ax.bar(x, shared, color=IEEE_COLORS["normal"], label="shared")
    ax.bar(x, so, bottom=shared, color=IEEE_COLORS["residual"], label="source-only")
    ax.bar(x, to, bottom=shared + so, color=IEEE_COLORS["raw"], label="target-only")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("CAN IDs")
    ax.set_title(title)
    ax.legend(fontsize=6)
    return save_ieee_figure(fig, path)


def plot_smd(shift_df: pd.DataFrame, path: str | Path, *, title: str, top: int = 12) -> Path:
    """Top-|SMD| feature shifts between source and target normals."""
    apply_ieee_style()
    if shift_df.empty:
        fig, ax = plt.subplots(figsize=figsize_single(2.4))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return save_ieee_figure(fig, path)
    d = shift_df.dropna(subset=["smd"]).copy()
    d["abs"] = d["smd"].abs()
    d = d.sort_values("abs", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize_single(3.2))
    colors = [IEEE_COLORS["attack"] if v > 0 else IEEE_COLORS["raw"] for v in d["smd"]]
    ax.barh(d["feature"], d["smd"], color=colors)
    ax.axvline(0, color=IEEE_COLORS["neutral"], lw=0.8)
    ax.set_xlabel("SMD (target − source)")
    ax.set_title(title)
    return save_ieee_figure(fig, path)


def plot_f1_bars(df: pd.DataFrame, name_col: str, path: str | Path, *, title: str) -> Path:
    """Grouped PIRD/global/hybrid F1 bars per transfer row."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=figsize_double(2.6))
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return save_ieee_figure(fig, path)
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df.get("primary_f1", pd.Series([np.nan] * len(df))), width,
           label="PIRD (known-ID)", color=IEEE_COLORS["residual"])
    ax.bar(x + width / 2, df.get("fallback_f1", pd.Series([np.nan] * len(df))), width,
           label="global-fallback (unseen-ID)", color=IEEE_COLORS["raw"])
    ax.set_xticks(x)
    ax.set_xticklabels(df[name_col].astype(str), rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(fontsize=6)
    return save_ieee_figure(fig, path)


def plot_per_id_smd(df: pd.DataFrame, path: str | Path, *, title: str,
                    source_col: str = "source_domain", target_col: str = "target_domain") -> Path:
    """Mean |SMD| per shared CAN ID, one group per source→target transfer."""
    apply_ieee_style()
    if df.empty or "mean_abs_smd" not in df.columns:
        fig, ax = plt.subplots(figsize=figsize_single(2.4))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return save_ieee_figure(fig, path)
    d = df.copy()
    d["transfer"] = d[source_col].astype(str) + "\u2192" + d[target_col].astype(str)
    labels = [f"{cid} ({tr})" for cid, tr in zip(d["can_id"].astype(str), d["transfer"])]
    fig, ax = plt.subplots(figsize=figsize_single(3.4))
    ax.barh(labels, d["mean_abs_smd"], color=IEEE_COLORS["attack"])
    ax.set_xlabel("mean |SMD| across features")
    ax.set_title(title)
    return save_ieee_figure(fig, path)


def plot_operating_curve(df: pd.DataFrame, path: str | Path, *, title: str) -> Path:
    """Recall vs actual target FPR for source-selected thresholds."""
    apply_ieee_style()
    if df.empty or "actual_fpr" not in df.columns or "recall" not in df.columns:
        fig, ax = plt.subplots(figsize=figsize_single(2.4))
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return save_ieee_figure(fig, path)
    d = df.dropna(subset=["actual_fpr", "recall"])
    fig, ax = plt.subplots(figsize=figsize_single(2.8))
    ax.plot(d["actual_fpr"], d["recall"], marker="o", color=IEEE_COLORS["residual"])
    for _, r in d.iterrows():
        ax.annotate(f"{r.get('target_fpr', float('nan')):g}", (r["actual_fpr"], r["recall"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("actual target FPR")
    ax.set_ylabel("recall")
    ax.set_title(title)
    return save_ieee_figure(fig, path)


__all__ = [
    "plot_f1_bars",
    "plot_id_overlap",
    "plot_operating_curve",
    "plot_per_id_smd",
    "plot_smd",
    "plot_transfer_heatmap",
]
