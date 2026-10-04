"""Phase 4B tests: representation-density scorers, shift diagnostics, probes."""

from __future__ import annotations

import numpy as np
import torch

from canguard.foundation.density import (
    apply_standardizer,
    distance_ratio,
    fit_centroid,
    fit_knn_reference,
    fit_mahalanobis,
    fit_standardizer,
    linear_probe,
    mmd_rbf,
    representation_shift,
    score_centroid,
    score_knn,
    score_mahalanobis,
)
from canguard.foundation.model import CANTFMModel
from canguard.foundation.scoring import embed_positions
from canguard.foundation.tokenizer import TokenizerConfig


def _blob(n, mean, std=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=(n, 8)).astype(np.float64)


def test_mahalanobis_diag_and_full_rank_far_point_higher():
    Z = _blob(3000, 0.0, seed=0)
    for diag in (True, False):
        stats = fit_mahalanobis(Z, shrinkage=0.1, diagonal=diag)
        near = score_mahalanobis(np.zeros((1, 8)), stats)
        far = score_mahalanobis(np.full((1, 8), 6.0), stats)
        assert far[0] > near[0]


def test_centroid_distance():
    Z = _blob(1000, 1.0, seed=1)
    mu = fit_centroid(Z)
    s = score_centroid(Z, mu)
    assert s.shape == (1000,) and (s >= 0).all()


def test_knn_reference_subsample_and_score():
    Z = _blob(5000, 0.0, seed=2)
    refs = fit_knn_reference(Z, max_refs=500, seed=0)
    assert refs.shape == (500, 8)
    scores, q = score_knn(Z[:100], refs, k=5)
    assert scores.shape == (100,) and (scores >= 0).all()


def test_representation_shift_detects_shift():
    Zs = _blob(500, 0.0, seed=3)
    same = representation_shift(Zs, _blob(500, 0.0, seed=4))
    shifted = representation_shift(Zs, _blob(500, 3.0, seed=5))
    assert shifted["centroid_distance"] > same["centroid_distance"]
    assert shifted["mean_abs_smd"] > same["mean_abs_smd"]
    assert shifted["mmd_rbf"] > same["mmd_rbf"]


def test_distance_ratio_high_when_attacks_far():
    Zs = _blob(500, 0.0, seed=6)
    Zt_norm = _blob(500, 0.0, seed=7)
    Zt_atk = _blob(500, 8.0, seed=8)
    out = distance_ratio(Zs, Zt_norm, Zt_atk)
    assert out["attack_normal_distance_ratio"] > 2.0


def test_linear_probe_separable_and_not():
    rng = np.random.default_rng(0)
    Xn = rng.normal(0, 1, (300, 6))
    Xa = rng.normal(3, 1, (300, 6))
    Xtr, ytr = np.vstack([Xn, Xa]), np.r_[np.zeros(300, int), np.ones(300, int)]
    Xte, yte = (
        np.vstack([rng.normal(0, 1, (100, 6)), rng.normal(3, 1, (100, 6))]),
        np.r_[np.zeros(100, int), np.ones(100, int)],
    )
    out = linear_probe(Xtr, ytr, Xte, yte)
    assert out["probe_roc_auc"] > 0.9


def test_standardizer_source_only():
    Z = _blob(200, 5.0, std=2.0, seed=0)
    mu, sd = fit_standardizer(Z)
    Zs = apply_standardizer(Z, mu, sd)
    assert abs(Zs.mean()) < 1e-6
    assert abs(Zs.std() - 1.0) < 1e-3


def test_mmd_identical_is_small():
    Z = _blob(400, 0.0, seed=11)
    assert mmd_rbf(Z, Z + 1e-6, seed=0) < 1e-3


def test_embed_positions_causal_flag():
    spec = TokenizerConfig(id_mode="none")
    model = CANTFMModel(
        spec, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, backbone="transformer"
    )
    model.eval()
    X = np.random.rand(4, 8, spec.feature_dim).astype(np.float32)
    with torch.no_grad():
        Zc = embed_positions(model, X, causal=True)
        Zb = embed_positions(model, X, causal=False)
    assert Zc.shape == Zb.shape == (4, 8, 16)
    assert not np.allclose(Zc, Zb)
