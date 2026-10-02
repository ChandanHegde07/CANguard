
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

IEEE_COLORS = {
    "raw": "#0072B2",  # blue (colorblind-safe)
    "residual": "#E69F00",  # orange
    "normal": "#009E73",  # green
    "attack": "#D55E00",  # vermillion
    "grid": "#BBBBBB",
    "secondary": "#CC79A7",
    "neutral": "#333333",
}

# IEEEtran approximate figure widths (inches)
IEEE_SINGLE_COL = 3.5
IEEE_DOUBLE_COL = 7.16


def apply_ieee_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def figsize_single(height: float = 2.4) -> tuple[float, float]:
    return (IEEE_SINGLE_COL, height)


def figsize_double(height: float = 2.6) -> tuple[float, float]:
    return (IEEE_DOUBLE_COL, height)


def save_ieee_figure(fig: plt.Figure, path: str | Path, *, close: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    if close:
        plt.close(fig)
    return path
