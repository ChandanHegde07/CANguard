"""Evaluation metrics, threshold selection, and orchestration."""

from .bootstrap import bootstrap_metrics, flatten_bootstrap
from .deployment import (
    DEFAULT_FPR_TARGETS,
    DEFAULT_RECALL_TARGET,
    deployment_report,
    false_alarms_per_hour,
    flatten_deployment,
    fpr_at_recall,
    latency_report,
    recall_at_fpr,
    threshold_at_fpr,
    threshold_at_recall,
)
from .evaluation import cross_attack_evaluate, train_anomaly_detector
from .metrics import compute_metrics, confusion_table
from .threshold import choose_threshold_from_val_normals, sweep_thresholds

__all__ = [
    "DEFAULT_FPR_TARGETS",
    "DEFAULT_RECALL_TARGET",
    "bootstrap_metrics",
    "choose_threshold_from_val_normals",
    "compute_metrics",
    "confusion_table",
    "cross_attack_evaluate",
    "deployment_report",
    "false_alarms_per_hour",
    "flatten_bootstrap",
    "flatten_deployment",
    "fpr_at_recall",
    "latency_report",
    "recall_at_fpr",
    "sweep_thresholds",
    "threshold_at_fpr",
    "threshold_at_recall",
    "train_anomaly_detector",
]
