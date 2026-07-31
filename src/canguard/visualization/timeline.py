"""Timeline strip of anomaly scores with shaded attack regions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as label_blocks


def plot_score_timeline(
    scores: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    title: str = "Anomaly score vs time",
    figsize=(12, 3),
):
    """Plot score over window index, shading contiguous attack regions.

    Parameters
    ----------
    scores : np.ndarray
        Anomaly scores in chronological window order.
    y_test : np.ndarray
        Binary ground-truth labels.
    threshold : float
        Decision threshold drawn as a horizontal dashed line.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    scores = np.asarray(scores)
    y_test = np.asarray(y_test)
    ax.plot(scores, color="gray", lw=0.8, label="Anomaly score")
    if y_test.max() > 0:
        labeled, n_feat = label_blocks(y_test)
        for seg in range(1, n_feat + 1):
            idx = np.where(labeled == seg)[0]
            ax.axvspan(
                idx[0], idx[-1], alpha=0.15, color="crimson", label="Attack" if seg == 1 else ""
            )
    ax.axhline(threshold, color="k", ls="--", lw=1, label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
