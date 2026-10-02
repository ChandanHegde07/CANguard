"""Phase 2 adaptive-attack figures (detection boundary, drift, latencies)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import IEEE_COLORS, apply_ieee_style, figsize_double, figsize_single, save_ieee_figure


def plot_detection_probability(
    boundary_df: pd.DataFrame,
    path: str | Path,
    *,
    severity_col: str = "severity",
    prob_col: str = "detection_probability",
    group_col: str | None = None,
    title: str = "Adaptive attack detection boundary",
) -> Path:
    """Detection probability vs attack severity alpha, with Wilson CI bands."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=figsize_single(2.6))
    df = boundary_df.copy()
    groups = (
        [(None, df)]
        if not group_col or group_col not in df.columns
        else list(df.groupby(group_col))
    )
    palette = [
        IEEE_COLORS["residual"],
        IEEE_COLORS["raw"],
        IEEE_COLORS["normal"],
        IEEE_COLORS["secondary"],
    ]
    for i, (gname, g) in enumerate(groups):
        g = g.sort_values(severity_col)
        x = g[severity_col].to_numpy(dtype=float)
        y = g[prob_col].to_numpy(dtype=float)
        color = palette[i % len(palette)]
        label = str(gname) if gname is not None else "pooled"
        ax.plot(x, y, marker="o", color=color, label=label)
        lo = g.get("detection_prob_ci_low")
        hi = g.get("detection_prob_ci_high")
        if lo is not None and hi is not None:
            ax.fill_between(
                x,
                lo.to_numpy(dtype=float),
                hi.to_numpy(dtype=float),
                color=color,
                alpha=0.15,
            )
    for level, ls in ((0.5, "--"), (0.9, ":")):
        ax.axhline(level, color=IEEE_COLORS["neutral"], ls=ls, lw=0.8, alpha=0.7)
    ax.set_xlabel(r"Attack severity $\alpha$")
    ax.set_ylabel("Detection probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    if group_col:
        ax.legend(title=group_col)
    return save_ieee_figure(fig, path)


def plot_residual_trajectory(
    traj_df: pd.DataFrame,
    path: str | Path,
    *,
    title: str = "Gradual-drift residual trajectory",
) -> Path:
    """Two-panel gradual-drift trajectory: residual norm/alpha and score/threshold."""
    apply_ieee_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize_double(4.2), sharex=True, gridspec_kw={"hspace": 0.12}
    )
    t = traj_df["t"].to_numpy(dtype=float)
    resid = traj_df["residual_norm"].to_numpy(dtype=float)
    ax_top.plot(t, resid, color=IEEE_COLORS["attack"], label="Residual norm")
    ax_top.set_ylabel("Residual L2 norm", color=IEEE_COLORS["attack"])
    ax_top.tick_params(axis="y", labelcolor=IEEE_COLORS["attack"])

    if "alpha" in traj_df.columns:
        ax_top2 = ax_top.twinx()
        ax_top2.plot(t, traj_df["alpha"].to_numpy(dtype=float), color=IEEE_COLORS["raw"],
                     ls="--", label=r"$\alpha$")
        ax_top2.set_ylabel(r"Attack severity $\alpha$", color=IEEE_COLORS["raw"])
        ax_top2.tick_params(axis="y", labelcolor=IEEE_COLORS["raw"])
        ax_top2.grid(False)

    if "score" in traj_df.columns:
        scores = traj_df["score"].to_numpy(dtype=float)
        ax_bot.plot(t, scores, color=IEEE_COLORS["secondary"], label="Anomaly score")
        if "threshold" in traj_df.columns:
            thr = float(traj_df["threshold"].iloc[0])
            ax_bot.axhline(thr, color=IEEE_COLORS["neutral"], ls="--", lw=0.9,
                           label="score threshold")
        if "flagged" in traj_df.columns:
            flagged = traj_df["flagged"].to_numpy(dtype=bool)
            ax_bot.scatter(t[flagged], scores[flagged], s=6, color=IEEE_COLORS["attack"],
                           zorder=3, label="flagged")
    ax_bot.set_xlabel("Window index within horizon")
    ax_bot.set_ylabel("Anomaly score")

    for ax in (ax_top, ax_bot):
        ax.axvline(0, color=IEEE_COLORS["neutral"], ls=":", lw=1.0)
        if "flagged" in traj_df.columns:
            flagged = traj_df["flagged"].to_numpy(dtype=bool)
            if flagged.any():
                ax.axvline(int(np.argmax(flagged)), color=IEEE_COLORS["normal"], ls="-.", lw=1.0)
    ax_top.plot([], [], color=IEEE_COLORS["neutral"], ls=":", label="attack start (x=0)")
    if "flagged" in traj_df.columns and traj_df["flagged"].any():
        ax_top.plot([], [], color=IEEE_COLORS["normal"], ls="-.", label="first alarm")

    ax_top.set_title(title)
    ax_top.legend(loc="upper left", fontsize=6)
    ax_bot.legend(loc="upper left", fontsize=6)
    return save_ieee_figure(fig, path)


def plot_residual_distributions(
    normal_norm: np.ndarray,
    attack_norm: np.ndarray,
    path: str | Path,
    *,
    label: str = "adaptive attack",
    title: str = "Residual-norm distributions",
) -> Path:
    """Normal vs adaptive-attack residual L2-norm histograms."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=figsize_single(2.6))
    bins = np.linspace(
        0.0,
        float(max(np.max(normal_norm), np.max(attack_norm), 1e-6)),
        50,
    )
    ax.hist(
        normal_norm, bins=bins, density=True, alpha=0.55,
        color=IEEE_COLORS["normal"], label="Normal",
    )
    ax.hist(
        attack_norm, bins=bins, density=True, alpha=0.55,
        color=IEEE_COLORS["attack"], label=label,
    )
    ax.set_xlabel("Residual L2 norm")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return save_ieee_figure(fig, path)


def plot_latency_vs_severity(
    latency_df: pd.DataFrame,
    path: str | Path,
    *,
    severity_col: str = "severity",
    latency_col: str = "delay_windows",
    group_col: str = "rate",
    title: str = "Detection latency vs attack severity",
) -> Path:
    """Median detection latency vs end severity, one line per drift rate."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=figsize_single(2.6))
    df = latency_df.dropna(subset=[latency_col]).copy()
    if df.empty:
        ax.text(0.5, 0.5, "no detections", ha="center", va="center", transform=ax.transAxes)
    else:
        groups = list(df.groupby(group_col)) if group_col in df.columns else [("all", df)]
        palette = [IEEE_COLORS["normal"], IEEE_COLORS["residual"], IEEE_COLORS["attack"]]
        for i, (gname, g) in enumerate(groups):
            agg = g.groupby(severity_col)[latency_col].agg(["median", "count"])
            q25 = g.groupby(severity_col)[latency_col].quantile(0.25).to_numpy(dtype=float)
            q75 = g.groupby(severity_col)[latency_col].quantile(0.75).to_numpy(dtype=float)
            med = agg["median"].to_numpy(dtype=float)
            ax.errorbar(
                agg.index.to_numpy(dtype=float),
                med,
                yerr=np.vstack([med - q25, q75 - med]),
                marker="o",
                capsize=2,
                color=palette[i % len(palette)],
                label=str(gname),
            )
    ax.set_xlabel(r"Final attack severity $\alpha_{end}$")
    ax.set_ylabel("Detection latency (windows)")
    ax.set_title(title)
    if group_col in df.columns:
        ax.legend(title=group_col)
    return save_ieee_figure(fig, path)


__all__ = [
    "plot_detection_probability",
    "plot_latency_vs_severity",
    "plot_residual_distributions",
    "plot_residual_trajectory",
]
