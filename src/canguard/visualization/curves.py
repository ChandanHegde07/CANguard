"""ROC / PR curve visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_roc_pr(
    y_test: np.ndarray,
    scores: np.ndarray,
    fpr_at_op: float | None = None,
    title_prefix: str = "",
    figsize=(10, 4),
):
    """Plot ROC and PR curves side by side with the operating point marked.

    Parameters
    ----------
    y_test : np.ndarray
        Binary ground-truth labels.
    scores : np.ndarray
        Continuous anomaly scores.
    fpr_at_op : float | None
        If given, plots a marker on the ROC curve at the achieved FPR.
    title_prefix : str
        Prefix added to subplot titles.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_test = np.asarray(y_test)
    scores = np.asarray(scores)
    try:
        roc_auc = roc_auc_score(y_test, scores)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_test, scores)
    except Exception:
        pr_auc = float("nan")

    fpr, tpr, _ = roc_curve(y_test, scores)
    prec, rec, _ = precision_recall_curve(y_test, scores)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    if fpr_at_op is not None:
        idx = int(np.argmin(np.abs(fpr - fpr_at_op)))
        ax.scatter(
            fpr[idx],
            tpr[idx],
            color="crimson",
            zorder=5,
            s=50,
            label=f"Op point (FPR={fpr_at_op:.3f})",
        )
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{title_prefix}ROC Curve".strip())
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(rec, prec, color="crimson", lw=2, label=f"PR (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix}PR Curve".strip())
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_threshold_sweep(sweep_rows: list[dict], title: str = "", figsize=(6, 4), ax=None):
    """Plot recall and actual-FPR vs target-FPR from a threshold sweep.

    Parameters
    ----------
    sweep_rows : list[dict]
        Rows from :func:`canguard.evaluation.threshold.sweep_thresholds` with
        keys target_FPR, recall, actual_FPR.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes | None

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    targets = [r["target_FPR"] for r in sweep_rows]
    recalls = [r["recall"] for r in sweep_rows]
    actuals = [r["actual_FPR"] for r in sweep_rows]
    ax.plot(targets, recalls, "o-", color="steelblue", label="Recall")
    ax.plot(targets, actuals, "s--", color="crimson", label="Actual FPR")
    ax.set_xscale("log")
    ax.set_xlabel("Target FPR")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax
