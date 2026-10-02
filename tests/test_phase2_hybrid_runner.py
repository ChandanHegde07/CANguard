"""End-to-end smoke test for the Phase 2 hybrid (global + PIRD) runner."""

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
        attack = cid == "0000" and i >= 200
        rows.append([f"{ts:.6f}", cid, "8", *data, "T" if attack else "R"])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _config(tmp_path: Path) -> dict:
    return {
        "phase": 2,
        "experiment_id": "phase2_hybrid_smoke",
        "seed": 0,
        "dataset": "hcrl",
        "data_dir": str(tmp_path / "data"),
        "datasets": ["RPM"],
        "sample_size": 1500,
        "window_size": 5,
        "split": {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4},
        "detector": {"kind": "isolation_forest", "n_estimators": 10, "random_state": 0},
        "evaluation": {"val_holdout_fraction": 0.2, "fpr_target": 0.01},
        "global": {
            "window_frames": 10,
            "normalization": "robust",
            "id_activity_frac": 0.05,
            "run_ablation": True,
            "detector": {"kind": "isolation_forest", "n_estimators": 10, "random_state": 0},
        },
        "hybrid": {"lambdas": [0.25, 0.5, 0.75], "ablation_severities": [0.5, 1.0]},
        "targeting": {"top_k": 2, "min_calib_windows": 5, "max_calib_attack_frac": 0.0,
                      "target_ids": None},
        "attack": {
            "type": "adaptive_legitimate_id",
            "target_features": "all_attackable",
            "severities": [0.5, 1.0],
            "directions": ["positive"],
            "attack_modes": ["additive"],
            "example_severity": 1.0,
            "drift": {"rates": ["fast"], "shape": "linear", "horizon_windows": 10,
                      "fractions": None, "save_trajectories": True},
        },
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "phase2"),
        "figures_dir": str(tmp_path / "figures"),
    }


def test_hybrid_runner_smoke(tmp_path: Path) -> None:
    from canguard.visualization.hybrid import make_hybrid_plots
    from experiments.runners.phase2_hybrid import run_hybrid

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_tiny_hcrl_csv(data_dir / "RPM_dataset.csv")

    cfg = _config(tmp_path)
    out_root = tmp_path / "phase2"
    result = run_hybrid(cfg, out_root)

    assert len(result["conventional"]) > 0
    assert {"pird", "global", "max", "or"} <= set(result["conventional"]["rule"])
    assert (out_root / "hybrid" / "conventional_metrics.csv").exists()
    assert (out_root / "hybrid" / "fixed_fpr.csv").exists()
    assert (out_root / "hybrid" / "adaptive_boundary.csv").exists()
    assert (out_root / "hybrid" / "complementarity.csv").exists()
    assert (out_root / "global" / "conventional_metrics.csv").exists()
    assert (out_root / "hybrid" / "scores_RPM.csv").exists()

    adaptive = result["adaptive"]
    assert {"pird", "global", "max"} <= set(adaptive["rule"])
    # Global ablation families present.
    assert adaptive["rule"].str.startswith("global_").any()

    make_hybrid_plots(cfg, out_root, result)
    assert (out_root / "plots" / "hybrid_detection_probability_vs_severity.png").exists()
    assert (out_root / "plots" / "hybrid_complementarity.png").exists()


def test_hybrid_runner_reproducible(tmp_path: Path) -> None:
    from experiments.runners.phase2_hybrid import run_hybrid

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_tiny_hcrl_csv(data_dir / "RPM_dataset.csv")
    cfg = _config(tmp_path)
    cfg["sample_size"] = 700
    a = run_hybrid(cfg, tmp_path / "run_a")
    b = run_hybrid(cfg, tmp_path / "run_b")
    key = ["dataset", "rule", "fpr_target"]
    a_conv = a["conventional"][key + ["recall", "actual_fpr"]].reset_index(drop=True)
    b_conv = b["conventional"][key + ["recall", "actual_fpr"]].reset_index(drop=True)
    import pandas as pd

    pd.testing.assert_frame_equal(a_conv, b_conv)
