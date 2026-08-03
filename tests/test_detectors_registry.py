"""Smoke tests for multi-detector registry and interface."""

from __future__ import annotations

import numpy as np
import pytest

from canguard.detectors import create_detector, list_detectors


def _data(seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, size=(200, 6))
    attack = rng.normal(5, 1, size=(40, 6))
    return normal, attack


@pytest.mark.parametrize("kind", list_detectors())
def test_detector_fit_score_predict_save(kind, tmp_path):
    normal, attack = _data()
    det = create_detector(kind, random_state=0) if kind != "hbos" else create_detector(kind)
    det.fit(normal)
    scores_n = det.score(normal[:20])
    scores_a = det.score(attack[:20])
    assert scores_n.shape == (20,)
    assert np.all(np.isfinite(scores_n))
    thr = float(np.percentile(scores_n, 99))
    det.set_threshold(thr)
    preds = det.predict(np.vstack([normal[:20], attack[:20]]))
    assert preds.shape == (40,)
    path = tmp_path / f"{kind}.joblib"
    det.save(path)
    loaded = type(det).load(path)
    assert np.allclose(loaded.score(normal[:10]), det.score(normal[:10]), rtol=1e-4, atol=1e-4)
