"""Tests for the evaluation orchestration and cross-attack protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd

from canguard.detectors import IsolationForestDetector
from canguard.evaluation import cross_attack_evaluate, train_anomaly_detector
from canguard.features import fit_per_id_stats, transform_residuals
from canguard.features.groups import BEHAVIORAL_FEATURES_V1


def _synth(seed=0, n_norm=120, n_att=20):
    rng = np.random.default_rng(seed)
    ids = ["0316", "018f", "02b0"]
    feat_mu = {"0316": 1.0, "018f": 2.0, "02b0": 3.0}
    rows = []
    ts = 1000.0
    for _ in range(n_norm):
        cid = ids[rng.integers(0, 3)]
        r = {"can_id": cid, "timestamp": ts, "is_attack": 0, "attack_frac": 0.0}
        for f in BEHAVIORAL_FEATURES_V1:
            r[f] = float(rng.normal(feat_mu[cid], 1.0))
        rows.append(r)
        ts += 0.001
    for _ in range(n_att):
        cid = "0000"  # novel ID -> global fallback
        r = {"can_id": cid, "timestamp": ts, "is_attack": 1, "attack_frac": 1.0}
        for f in BEHAVIORAL_FEATURES_V1:
            r[f] = float(rng.normal(20.0, 1.0))
        rows.append(r)
        ts += 0.001
    return pd.DataFrame(rows)


def test_train_anomaly_detector_protocol():
    df = _synth(seed=0)
    stats, gstats = fit_per_id_stats(df, BEHAVIORAL_FEATURES_V1)
    res = transform_residuals(df, stats, gstats, BEHAVIORAL_FEATURES_V1)
    res_cols = [c + "_res" for c in BEHAVIORAL_FEATURES_V1]
    split = int(len(res) * 0.5)
    train_df, test_df = res.iloc[:split], res.iloc[split:]

    det = IsolationForestDetector(n_estimators=30, random_state=0, n_jobs=1)
    out = train_anomaly_detector(det, train_df, test_df, res_cols)
    assert "recall" in out and "fpr" in out and "f1" in out
    assert "scores_test" in out and "y_test" in out and "model" in out
    assert len(out["scores_test"]) == len(test_df)
    # Novel-ID attack cluster should be heavily flagged
    assert out["recall"] > 0.5


def test_train_anomaly_detector_normal_only():
    df = _synth(seed=1)
    stats, gstats = fit_per_id_stats(df, BEHAVIORAL_FEATURES_V1)
    res = transform_residuals(df, stats, gstats, BEHAVIORAL_FEATURES_V1)
    res_cols = [c + "_res" for c in BEHAVIORAL_FEATURES_V1]
    split = int(len(res) * 0.5)
    train_df, test_df = res.iloc[:split], res.iloc[split:]
    det = IsolationForestDetector(n_estimators=30, random_state=0, n_jobs=1)
    out = train_anomaly_detector(det, train_df, test_df, res_cols)

    n_normals = int((train_df["is_attack"] == 0).sum())
    n_val = max(1, int(n_normals * 0.2))
    expected_fit = n_normals - n_val if n_val < n_normals else n_normals
    # Detector is fit on all train normals except the last-20% validation holdout.
    assert out["n_train_normals"] == expected_fit


def test_cross_attack_evaluate():
    src = _synth(seed=2, n_norm=120, n_att=0)  # no attack in source
    tgt = _synth(seed=3, n_norm=60, n_att=20)  # has attacks
    # Fit target stats (single residualization discipline: target stats for both)
    stats_t, gstats_t = fit_per_id_stats(tgt, BEHAVIORAL_FEATURES_V1)
    src_res = transform_residuals(src, stats_t, gstats_t, BEHAVIORAL_FEATURES_V1)
    tgt_res = transform_residuals(tgt, stats_t, gstats_t, BEHAVIORAL_FEATURES_V1)
    res_cols = [c + "_res" for c in BEHAVIORAL_FEATURES_V1]
    src_normals = src_res[src_res["is_attack"] == 0]
    det = IsolationForestDetector(n_estimators=30, random_state=0, n_jobs=1)
    out = cross_attack_evaluate(det, src_normals, tgt_res, res_cols)
    assert out["recall"] > 0.5
    assert out["fpr"] >= 0.0
