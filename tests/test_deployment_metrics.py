"""Tests for deployment-oriented metrics (Phase 1)."""

from __future__ import annotations

import numpy as np

from canguard.evaluation import (
    deployment_report,
    false_alarms_per_hour,
    flatten_deployment,
    fpr_at_recall,
    latency_report,
    recall_at_fpr,
    threshold_at_fpr,
    threshold_at_recall,
)


def test_threshold_at_fpr_matches_percentile():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, 10_000)
    th = threshold_at_fpr(normal, 0.01)
    assert abs((normal >= th).mean() - 0.01) < 0.01


def test_recall_at_fpr_on_separable_scores():
    rng = np.random.default_rng(1)
    normal = rng.normal(0, 1, 5_000)
    scores = np.concatenate([normal, rng.normal(6, 1, 500)])
    y = np.concatenate([np.zeros(5_000), np.ones(500)]).astype(int)
    row = recall_at_fpr(normal, y, scores, 0.01)
    assert row["recall"] > 0.99
    assert row["actual_fpr"] < 0.02


def test_threshold_at_recall_meets_target():
    rng = np.random.default_rng(2)
    n_pos = 400
    scores = np.concatenate([rng.normal(0, 1, 4_000), rng.normal(2, 1, n_pos)])
    y = np.concatenate([np.zeros(4_000), np.ones(n_pos)]).astype(int)
    th = threshold_at_recall(y, scores, 0.95)
    achieved = (scores[y == 1] >= th).mean()
    assert achieved >= 0.95


def test_fpr_at_recall_reports_operating_point():
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(0, 1, 4_000), rng.normal(3, 1, 400)])
    y = np.concatenate([np.zeros(4_000), np.ones(400)]).astype(int)
    row = fpr_at_recall(y, scores, 0.90)
    assert row["recall"] >= 0.90
    assert 0.0 <= row["fpr"] <= 1.0


def test_false_alarms_per_hour_scales_with_duration():
    # 7200 windows over exactly one hour (0.5 s apart on average).
    timestamps = np.linspace(0.0, 3600.0, 7200)
    y_true = np.zeros(7200, dtype=int)
    y_pred = np.zeros(7200, dtype=int)
    y_pred[:10] = 1  # 10 false alarms in one hour
    out = false_alarms_per_hour(timestamps, y_true, y_pred)
    assert out["n_false_alarms"] == 10
    assert abs(out["observed_hours"] - 1.0) < 1e-6
    assert abs(out["false_alarms_per_hour"] - 10.0) < 1e-6


def test_latency_report_averages_segment_delays():
    timestamps = np.arange(20, dtype=float) * 0.01  # 10 ms per window
    y_true = np.zeros(20, dtype=int)
    y_true[5:10] = 1
    y_true[15:19] = 1
    y_pred = np.zeros(20, dtype=int)
    y_pred[7] = 1   # 2 windows after first segment start
    y_pred[17] = 1  # 2 windows after second segment start
    out = latency_report(timestamps, y_true, y_pred)
    assert out["n_attack_segments"] == 2
    assert out["n_detected_segments"] == 2
    assert abs(out["mean_delay_ms"] - 20.0) < 1e-6


def test_deployment_report_and_flatten_have_expected_keys():
    rng = np.random.default_rng(4)
    normal = rng.normal(0, 1, 2_000)
    scores = np.concatenate([normal, rng.normal(4, 1, 200)])
    y = np.concatenate([np.zeros(2_000), np.ones(200)]).astype(int)
    y_pred = np.concatenate([np.zeros(2_000), np.ones(200)]).astype(int)
    timestamps = np.linspace(0, 1000, len(y))
    report = deployment_report(normal, timestamps, y, scores, y_pred,
                               fpr_targets=(0.01,), recall_target=0.95)
    flat = flatten_deployment(report)
    assert "recall_at_fpr_0.01" in flat
    assert "fpr_at_recall" in flat
    assert "false_alarms_per_hour" in flat
    assert "mean_delay_ms" in flat
