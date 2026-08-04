
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _style_ax(ax, title: str, xlabel: str = "Anomaly score") -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")


def plot_score_distribution(
    scores: np.ndarray,
    y_test: np.ndarray,
    threshold: float | None = None,
    title: str = "Score distribution",
    ax=None,
):

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    y_test = np.asarray(y_test)
    scores = np.asarray(scores)
    ax.hist(
        scores[y_test == 0],
        bins=50,
        alpha=0.5,
        label="Normal",
        color="steelblue",
        density=True,
    )
    ax.hist(
        scores[y_test == 1],
        bins=50,
        alpha=0.5,
        label="Attack",
        color="crimson",
        density=True,
    )
    if threshold is not None:
        ax.axvline(threshold, color="k", ls="--", lw=1.5, label=f"thresh={threshold:.2f}")
    ax.legend(fontsize=8)
    _style_ax(ax, title)
    return ax


def plot_score_distribution_grid(results: dict, names: list[str], ncols: int = 2, figsize=(12, 8)):

    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    for ax, name in zip(axes, names):
        r = results[name]
        plot_score_distribution(r["scores_test"], r["y_test"], r["threshold"], title=name, ax=ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
