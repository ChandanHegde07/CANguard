"""Equivalence/behavior tests for the IsolationForest detector."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF

from canguard.detectors import IsolationForestDetector


def _make_normal_anomalous(seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, size=(200, 4))
    anomalous = rng.normal(8, 1, size=(50, 4))  # far from normal mass
    return normal, anomalous


def test_fit_and_score_signature():
    normal, _ = _make_normal_anomalous()
    det = IsolationForestDetector(n_estimators=50, random_state=0, n_jobs=1)
    det.fit(normal)
    X = np.vstack([normal[:20], _make_normal_anomalous()[1][:5]])
    scores = det.score_samples(X)
    assert scores.shape == (25,)
    assert np.all(np.isfinite(scores))


def test_scores_match_sklearn_convention():
    normal, _ = _make_normal_anomalous(seed=1)
    det = IsolationForestDetector(n_estimators=100, random_state=0, n_jobs=1)
    det.fit(normal)
    sklearn_if = SklearnIF(n_estimators=100, random_state=0, n_jobs=1, contamination="auto")
    sklearn_if.fit(normal)
    X = normal
    assert np.allclose(det.score_samples(X), -sklearn_if.score_samples(X))


def test_anomalous_scores_higher_than_normal():
    normal, anomalous = _make_normal_anomalous(seed=2)
    det = IsolationForestDetector(n_estimators=100, random_state=0, n_jobs=1)
    det.fit(normal)
    n_s = det.score_samples(normal[:50])
    a_s = det.score_samples(anomalous[:50])
    assert float(a_s.mean()) > float(n_s.mean())


def test_predict_with_threshold():
    normal, anomalous = _make_normal_anomalous(seed=3)
    det = IsolationForestDetector(n_estimators=100, random_state=0, n_jobs=1)
    det.fit(normal)
    X = np.vstack([normal[:50], anomalous[:50]])
    scores = det.score_samples(X)
    th = np.percentile(scores[:50], 99)  # ~1% FPR on the normal subset
    preds = det.predict_with_threshold(X, th)
    assert preds[:50].sum() <= 1
    assert preds[50:].sum() > 0  # some anomalous flagged


def test_feature_importances_none():
    det = IsolationForestDetector()
    assert det.feature_importances_ is None


def test_seed_reproducibility():
    normal, _ = _make_normal_anomalous(seed=4)
    a = IsolationForestDetector(n_estimators=100, random_state=0, n_jobs=1)
    b = IsolationForestDetector(n_estimators=100, random_state=0, n_jobs=1)
    a.fit(normal)
    b.fit(normal)
    assert np.allclose(a.score_samples(normal), b.score_samples(normal))
