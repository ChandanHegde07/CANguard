"""Phase 4 tests: tokenizer, ID-agnostic representation, model, leakage, scoring,
zero-shot metrics, checkpointing, and a tiny end-to-end pipeline run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from canguard.evaluation.zero_shot import (
    evaluate_zero_shot_scores,
    sustained_detection_delay,
    threshold_from_normals,
    tpr_at_fpr,
)
from canguard.foundation.checkpoint import load_checkpoint, save_checkpoint
from canguard.foundation.data import list_captures
from canguard.foundation.leakage import (
    LeakageError,
    assert_disjoint,
    assert_no_target_fit,
    assert_normal_windows,
)
from canguard.foundation.model import CANTFMModel
from canguard.foundation.objectives import next_event_loss
from canguard.foundation.pipeline import SourceModel, evaluate_source_on_target, train_source
from canguard.foundation.scoring import fit_residual_weights, score_next_event
from canguard.foundation.tokenizer import TokenizerConfig, frames_to_features, make_windows
from canguard.foundation.train import TrainConfig


def _frames(n=400, ids=("0316", "018f", "0000"), attack_from=300):
    rows = []
    for i in range(n):
        cid = ids[i % len(ids)]
        data = [f"{(i + j) % 256:02X}" for j in range(8)]
        rows.append(
            {
                "timestamp": 1000.0 + i * 0.001,
                "can_id": cid,
                "dlc": 8,
                **{f"data_{j}": data[j] for j in range(8)},
                "is_attack": int(i >= attack_from and cid == "0000"),
                "attack_type": "DoS" if (i >= attack_from and cid == "0000") else "Normal",
                "domain": "synth",
                "capture": "capA",
            }
        )
    return pd.DataFrame(rows)


# --- tokenizer / representation -------------------------------------------
def test_feature_dims_by_id_mode():
    df = _frames(50)
    for id_mode, expected in [("none", 10), ("numeric", 11), ("numeric_embed", 11), ("bit", 21)]:
        spec = TokenizerConfig(payload_mode="byte", id_mode=id_mode, id_bits=11)
        X, slices, valid = frames_to_features(df, spec)
        assert X.shape == (50, expected)
        assert slices["payload"] == (0, 8)
        assert slices["dlc"] == (8, 9)
        assert slices["timing"] == (9, 10)
    bit_spec = TokenizerConfig(payload_mode="bit", id_mode="none", id_bits=11)
    Xb, sb, _ = frames_to_features(df, bit_spec)
    assert Xb.shape == (50, 66) and sb["payload"] == (0, 64)


def test_bit_id_represents_unseen_29bit_ids():
    spec = TokenizerConfig(id_mode="bit", id_bits=29)
    df = pd.DataFrame(
        {
            "timestamp": [1.0, 2.0],
            "can_id": ["18ff00f9", "18ff70f9"],
            "dlc": [8, 8],
            **{f"data_{i}": ["00", "01"] for i in range(8)},
        }
    )
    X, slices, _ = frames_to_features(df, spec)
    is_, ie = slices["id"]
    assert X[0, is_:ie].sum() > 0 and X[1, is_:ie].sum() > 0
    assert not np.array_equal(X[0, is_:ie], X[1, is_:ie])


def test_timing_feature_is_fixed_transform():
    df = _frames(5)
    spec = TokenizerConfig(id_mode="none")
    X, slices, _ = frames_to_features(df, spec)
    ts, te = slices["timing"]
    assert np.all((X[:, ts:te] >= 0) & (X[:, ts:te] <= 1))


def test_make_windows_shapes_and_no_crossing():
    feats = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    labels = np.zeros(40, dtype=int)
    X, y, starts = make_windows(feats, labels, length=10, stride=10)
    assert X.shape == (4, 10, 3) and y.shape == (4, 10)
    assert X[0, 0, 0] == 0 and X[1, 0, 0] == 30


# --- model / causality -----------------------------------------------------
@pytest.mark.parametrize("backbone", ["mlp", "gru", "transformer"])
def test_model_forward_shapes(backbone):
    spec = TokenizerConfig(id_mode="numeric", id_bits=11)
    model = CANTFMModel(
        spec, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, backbone=backbone, context=4
    )
    x = torch.rand(3, 16, spec.feature_dim)
    z, preds = model(x, causal=True)
    assert z.shape == (3, 16, 32)
    assert preds["payload"].shape == (3, 16, spec.payload_dim)
    assert preds["timing"].shape == (3, 16, 1)
    assert preds["id"].shape == (3, 16, 1)


@pytest.mark.parametrize("backbone", ["mlp", "gru", "transformer"])
def test_model_is_causal(backbone):
    spec = TokenizerConfig(id_mode="none")
    torch.manual_seed(0)
    model = CANTFMModel(
        spec, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, backbone=backbone, context=4
    )
    model.eval()
    x = torch.rand(2, 12, spec.feature_dim)
    _, p1 = model(x, causal=True)
    x2 = x.clone()
    x2[:, -1, :] = torch.rand_like(x2[:, -1, :])  # change the future
    _, p2 = model(x2, causal=True)
    # Early predictions must not depend on the final token.
    assert torch.allclose(p1["payload"][:, :10], p2["payload"][:, :10], atol=1e-5)


def test_next_event_loss_finite():
    spec = TokenizerConfig(id_mode="none")
    model = CANTFMModel(spec, d_model=32, nhead=2, num_layers=2, backbone="transformer")
    x = torch.rand(2, 10, spec.feature_dim)
    valid = torch.ones(2, 10, 8, dtype=torch.bool)
    _, preds = model(x, causal=True)
    p = spec.payload_dim
    slices = {"payload": (0, p), "timing": (p + 1, p + 2), "id": (p + 2, p + 2)}
    loss = next_event_loss(preds, x, valid, slices)
    assert torch.isfinite(loss)


# --- scoring ---------------------------------------------------------------
def test_scoring_shapes_and_weights():
    spec = TokenizerConfig(id_mode="none")
    model = CANTFMModel(spec, d_model=32, nhead=2, num_layers=2, backbone="gru")
    X = np.random.rand(4, 16, spec.feature_dim).astype(np.float32)
    valid = np.ones((4, 16, 8), dtype=bool)
    w = fit_residual_weights(model, X, valid, spec)
    assert w.shape == (spec.payload_dim + 1,)
    scores = score_next_event(model, X, valid, spec, w)
    assert scores.shape == (4, 15)
    assert np.isfinite(scores).all()


# --- leakage ---------------------------------------------------------------
def test_leakage_checks_raise():
    w = {"y": np.zeros((3, 10), dtype=int)}
    assert_normal_windows(w)  # no raise
    w["y"][1, 5] = 1
    with pytest.raises(LeakageError):
        assert_normal_windows(w)
    with pytest.raises(LeakageError):
        assert_disjoint(["a", "b"], ["b", "c"])
    with pytest.raises(LeakageError):
        assert_no_target_fit(["target"], ["source", "target"])


# --- zero-shot metrics -----------------------------------------------------
def test_threshold_and_tpr_at_fpr():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, 5000)
    thr = threshold_from_normals(normal, 0.01)
    assert thr > 2.0
    scores = np.concatenate([rng.normal(0, 1, 900), rng.normal(4, 1, 100)])
    labels = np.concatenate([np.zeros(900, int), np.ones(100, int)])
    diag = tpr_at_fpr(scores, labels, scores[labels == 0], 0.05)
    assert diag["tpr"] > 0.5 and diag["actual_fpr"] <= 0.1


def test_evaluate_zero_shot_uses_source_threshold():
    rng = np.random.default_rng(1)
    src = rng.normal(0, 1, 2000)
    tgt_norm = rng.normal(0, 1, 2000)
    tgt_atk = rng.normal(3, 1, 200)
    scores = np.concatenate([tgt_norm, tgt_atk])
    labels = np.concatenate([np.zeros(2000, int), np.ones(200, int)])
    res = evaluate_zero_shot_scores(src, scores, labels, fpr_target=0.01)
    assert res["recall"] > 0.5
    assert "tpr_at_0.01_fpr" in res and "tpr_at_0.05_fpr" in res


def test_sustained_detection_delay():
    times = np.arange(100, dtype=float) * 0.1
    labels = np.zeros(100, int)
    labels[40:60] = 1
    scores = np.zeros(100)
    scores[44:] = 10.0  # alarm starts 4 frames into the episode
    out = sustained_detection_delay(times, scores, labels, threshold=1.0, n_of_m=(3, 5))
    assert out["n_episodes"] == 1 and out["n_detected"] == 1
    assert out["median_delay_s"] > 0


# --- checkpoint ------------------------------------------------------------
def test_checkpoint_roundtrip(tmp_path: Path):
    spec = TokenizerConfig(id_mode="numeric", id_bits=11)
    model = CANTFMModel(
        spec, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, backbone="transformer"
    )
    model.eval()  # disable dropout so the comparison is deterministic
    path = save_checkpoint(tmp_path / "ckpt.pt", model, spec, {"k": 1})
    model2, spec2, meta = load_checkpoint(path)
    assert spec2 == spec and meta == {"k": 1}
    x = torch.rand(1, 8, spec.feature_dim)
    with torch.no_grad():
        _, p1 = model(x)
        _, p2 = model2(x)
    assert torch.allclose(p1["payload"], p2["payload"], atol=1e-6)


# --- tiny end-to-end pipeline on synthetic HCRL ----------------------------
def _write_tiny_hcrl(path: Path, n: int = 1600) -> None:
    """Two captures: a normal tail and an attack-containing head region."""
    ids = ["0316", "018f"]
    with open(path, "w", newline="") as f:
        for i in range(n):
            cid = ids[i % 2]
            ts = 1_500_000_000.0 + i * 0.001
            dlc = 8
            data = [f"{(i + j) % 256:02X}" for j in range(dlc)]
            attack = (i < n // 3) and (cid == "018f")  # attacks early (head)
            label = "T" if attack else "R"
            f.write(f"{ts:.6f},{cid},{dlc},{','.join(data)},{label}\n")


def test_pipeline_trains_and_scores_synthetic(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir()
    _write_tiny_hcrl(d / "DoS_dataset.csv")
    _write_tiny_hcrl(d / "Fuzzy_dataset.csv")
    roots = {"hcrl": str(d)}
    assert list_captures("hcrl", roots=roots) == ["DoS", "Fuzzy"]

    spec = TokenizerConfig(id_mode="none", id_bits=11)
    tr = TrainConfig(objective="next_event", epochs=1, batch_size=32, seed=0, device="cpu")
    model_cfg = {
        "model": {"backbone": "gru", "d_model": 32, "num_layers": 2, "dim_feedforward": 64}
    }
    src = train_source(
        ["hcrl"],
        spec,
        model_cfg,
        tr,
        length=16,
        roots=roots,
        max_frames_per_capture=1600,
        cache_dir=None,
        max_train_windows=500,
    )
    assert isinstance(src, SourceModel)
    from canguard.foundation.data import load_domain_frames

    frames = load_domain_frames(
        "hcrl", roots=roots, max_frames_per_capture=1600, tail_hcrl=False, cache=None
    )
    res = evaluate_source_on_target(src, frames, length=16)
    assert 0.0 <= res["target_actual_fpr"] <= 1.0
    assert "n_target" in res and res["n_target"] > 0
