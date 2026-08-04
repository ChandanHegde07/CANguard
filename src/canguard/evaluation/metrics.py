
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:

    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    try:
        auc = float(roc_auc_score(y_test, scores))
    except Exception:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_test, scores))
    except Exception:
        pr_auc = float("nan")

    return {
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def confusion_table(y_test: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return a labeled confusion-matrix table (Normal/Attack rows and cols)."""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["Normal", "Attack"], columns=["Pred Normal", "Pred Attack"])
