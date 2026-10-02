
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _annotate(ax, mat: np.ndarray) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = f"{val:.2f}" if val == val else "nan"
            color = "white" if (val == val and val > 0.6) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)


def plot_cross_attack_matrix(
    recall_matrix: pd.DataFrame,
    fpr_matrix: pd.DataFrame,
    figsize=(12, 5),
):

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    specs = [
        (axes[0], recall_matrix.values, "Recall (source -> target)", "RdYlGn", 0.0, 1.0),
        (axes[1], fpr_matrix.values, "FPR (source -> target)", "RdYlGn_r", 0.0, 1.0),
    ]
    for ax, mat, title, cmap, vmin, vmax in specs:
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(fpr_matrix.columns)))
        ax.set_xticklabels(fpr_matrix.columns, fontsize=9)
        ax.set_yticks(range(len(recall_matrix.index)))
        ax.set_yticklabels(recall_matrix.index, fontsize=9)
        ax.set_title(title, fontsize=10)
        _annotate(ax, mat)
        plt.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    return fig


def plot_ablation_heatmap(
    values: pd.DataFrame, title: str = "Feature-group ablation (F1)"
) -> plt.Figure:
    """Plot a small heatmap of F1 across feature groups x datasets."""
    mat = values.values
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(values.columns)))
    ax.set_xticklabels(values.columns)
    ax.set_yticks(range(len(values.index)))
    ax.set_yticklabels(values.index)
    ax.set_title(title)
    _annotate(ax, mat)
    plt.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    return fig
