"""Tests for corrected ROAD pre-injection residual protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from canguard.exp.road_protocol import (
    build_road_raw_residual_splits,
    build_road_window_table,
    list_eval_captures,
    resolve_road_root,
)


def _synth_frames(n=500, inject_start=2.0, inject_end=4.0, inject_id="0d0"):
    """Synthetic ROAD-like frames with elapsed-based injection on one ID."""
    rows = []
    ids = ["0d0", "033", "6e0"]
    ts0 = 1000.0
    for i in range(n):
        elapsed = i * 0.02  # 50 Hz
        cid = ids[i % 3]
        in_inj = inject_start <= elapsed <= inject_end and cid == inject_id
        rows.append(
            {
                "timestamp": ts0 + elapsed,
                "elapsed": elapsed,
                "can_id": cid,
                "dlc": 8,
                **{f"data_{k}": f"{(i + k) % 256:02x}" for k in range(8)},
                "is_attack": int(in_inj),
                "attack_type": "synth_attack" if in_inj else "R",
                "label": "T" if in_inj else "R",
            }
        )
    return pd.DataFrame(rows)


def test_pre_injection_stats_exclude_post_attack_normals():
    df = _synth_frames()
    ft = build_road_window_table(df, window_size=10)
    assert not ft.empty
    assert "elapsed" in ft.columns
    splits = build_road_raw_residual_splits(ft, [2.0, 4.0], min_pre_windows=20)
    assert "error" not in splits
    # Train is strictly pre-injection
    assert (splits["raw"]["train"]["elapsed"] < 2.0).all()
    # Test starts at injection
    assert (splits["raw"]["test"]["elapsed"] >= 2.0).all()
    # Residual train/test same lengths as raw
    assert len(splits["residual"]["train"]) == len(splits["raw"]["train"])
    assert len(splits["residual"]["test"]) == len(splits["raw"]["test"])
    assert splits["n_test_attack"] > 0


def test_resolve_road_root_nested(tmp_path: Path):
    root = tmp_path / "road" / "road"
    (root / "attacks").mkdir(parents=True)
    (root / "ambient").mkdir(parents=True)
    (root / "attacks" / "capture_metadata.json").write_text("{}")
    (root / "ambient" / "capture_metadata.json").write_text("{}")
    got = resolve_road_root(tmp_path / "road")
    assert got == root.resolve()


def test_list_eval_skips_masquerade_and_unlabeled(tmp_path: Path):
    root = tmp_path / "road"
    attacks = root / "attacks"
    attacks.mkdir(parents=True)
    (root / "ambient").mkdir()
    meta = {
        "good_attack_1": {"injection_id": "0xd0", "injection_interval": [1.0, 2.0]},
        "good_attack_1_masquerade": {"injection_id": "0xd0", "injection_interval": [1.0, 2.0]},
        "accelerator_x": {"injection_id": None, "injection_interval": None},
    }
    (attacks / "capture_metadata.json").write_text(json.dumps(meta))
    (root / "ambient" / "capture_metadata.json").write_text("{}")
    for name in meta:
        (attacks / f"{name}.log").write_text("(1000.0) can0 0d0#0011223344556677\n")
    items = list_eval_captures(root, skip_masquerade=True, skip_unlabeled=True)
    names = [n for n, _ in items]
    assert names == ["good_attack_1"]
