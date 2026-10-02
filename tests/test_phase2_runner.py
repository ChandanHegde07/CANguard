"""End-to-end smoke test for the Phase 2 runner on a tiny synthetic dataset.

Exercises baseline -> adaptive -> gradual through the same code paths used on
the real HCRL data, without the large CSVs.
"""

from __future__ import annotations

import csv
from pathlib import Path


def _write_tiny_hcrl_csv(path: Path, n_rows: int = 1500) -> None:
    rows = []
    ids = ["0316", "018f", "0000"]
    for i in range(n_rows):
        cid = ids[i % 3]
        ts = 1478191030.0 + i * 0.0005
        data = ["01"] * 8
        # Attacks on ID 0000 from ~13% onward, so calibration contains attack
        # windows for 0000 (which target selection must exclude).
        attack = cid == "0000" and i >= 200
        label = "T" if attack else "R"
        rows.append([f"{ts:.6f}", cid, "8", *data, label])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _base_config(tmp_path: Path) -> dict:
    return {
        "phase": 2,
        "experiment_id": "phase2_smoke",
        "seed": 0,
        "dataset": "hcrl",
        "data_dir": str(tmp_path / "data"),
        "datasets": ["RPM"],
        "sample_size": 1500,
        "window_size": 5,
        "split": {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4},
        "detector": {"kind": "isolation_forest", "n_estimators": 10, "random_state": 0},
        "evaluation": {"val_holdout_fraction": 0.2, "fpr_target": 0.01},
        "targeting": {"top_k": 2, "min_calib_windows": 5, "max_calib_attack_frac": 0.0,
                      "target_ids": None},
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "phase2"),
        "figures_dir": str(tmp_path / "figures"),
    }


def test_phase2_runner_baseline_adaptive_gradual(tmp_path: Path) -> None:
    from experiments.runners.run_phase2 import run_adaptive, run_baseline, run_gradual

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_tiny_hcrl_csv(data_dir / "RPM_dataset.csv")

    cfg = _base_config(tmp_path)
    cfg["attack"] = {
        "type": "adaptive_legitimate_id",
        "target_features": "all_attackable",
        "severities": [0.5, 1.0],
        "directions": ["positive", "negative"],
        "attack_modes": ["additive"],
        "example_severity": 1.0,
        "drift": {"rates": ["fast", "slow"], "shape": "linear", "horizon_windows": 10,
                  "fractions": None, "save_trajectories": True},
    }
    out_root = tmp_path / "phase2"

    base = run_baseline(cfg, out_root)
    assert len(base) == 1
    assert (out_root / "baseline" / "metrics.csv").exists()
    assert (out_root / "baseline" / "predictions_RPM.csv").exists()

    adaptive = run_adaptive(cfg, out_root)
    sweep = adaptive["sweep"]
    assert len(sweep) > 0
    assert "detection_probability" in sweep.columns
    # 0000 has calibrations attacks -> must not be targeted.
    assert set(sweep["target_id"].unique()) <= {"0316", "018f"}
    assert (out_root / "adaptive_attack" / "boundary_table.csv").exists()
    assert "alpha_50" in adaptive["summary"]["pooled"] or adaptive["summary"]["pooled"]

    gradual = run_gradual(cfg, out_root)
    res = gradual["results"]
    assert len(res) > 0
    assert (out_root / "gradual_drift" / "drift_results.csv").exists()
    assert (out_root / "gradual_drift" / "latency_table.csv").exists()
    # Trajectories grow monotonically in alpha per config.
    import pandas as pd

    traj = pd.read_csv(out_root / "gradual_drift" / "trajectories.csv")
    for _, g in traj.groupby(["target_id", "direction", "rate", "alpha_end"]):
        assert (g.sort_values("t")["alpha"].diff().dropna() >= -1e-12).all()
