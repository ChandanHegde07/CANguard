"""Evaluation metrics, threshold selection, and orchestration."""

from .bootstrap import bootstrap_metrics, flatten_bootstrap
from .evaluation import cross_attack_evaluate, train_anomaly_detector
from .metrics import compute_metrics, confusion_table
from .threshold import choose_threshold_from_val_normals, sweep_thresholds

__all__ = [
    "bootstrap_metrics",
    "choose_threshold_from_val_normals",
    "compute_metrics",
    "confusion_table",
    "cross_attack_evaluate",
    "flatten_bootstrap",
    "sweep_thresholds",
    "train_anomaly_detector",
]
