"""Evaluation metrics, threshold selection, and orchestration."""

from .evaluation import cross_attack_evaluate, train_anomaly_detector
from .metrics import compute_metrics, confusion_table
from .threshold import choose_threshold_from_val_normals, sweep_thresholds

__all__ = [
    "choose_threshold_from_val_normals",
    "compute_metrics",
    "confusion_table",
    "cross_attack_evaluate",
    "sweep_thresholds",
    "train_anomaly_detector",
]
