"""IEEE-ish matplotlib defaults for Phase C figures."""

from __future__ import annotations

import matplotlib as mpl


IEEE_COLORS = {
    "raw": "#1f77b4",
    "residual": "#ff7f0e",
    "normal": "#2ca02c",
    "attack": "#d62728",
    "grid": "#cccccc",
}


def apply_ieee_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )
