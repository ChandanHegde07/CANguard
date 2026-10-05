"""End-to-end smoke tests for the Phase 1 PIRD validation runner.

Real HCRL CSVs are git-ignored, so these tests generate tiny labelled streams in
the exact on-disk schema and exercise every experiment group plus the CLI driver
on them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def _write_stream(path: Path, n_rows: int = 5000, attack_id: str = "02B0") -> None:
    ids = ["0316", "018F", "02B0"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(n_rows):
            cid = ids[i % 3]
            ts = 1_478_000_000.0 + i * 0.0005
            label = "T" if (cid == attack_id and i > n_rows - 1000) else "R"
            writer.writerow([f"{ts:.6f}", cid, "8", *(["01"] * 8), label])


def _config(tmp_path: Path) -> dict:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_stream(data_dir / "RPM_dataset.csv", attack_id="02B0")
    _write_stream(data_dir / "gear_dataset.csv", attack_id="018F")
    return {
        "seed": 0,
        "data_dir": str(data_dir),
        "datasets": ["RPM", "gear"],
        "sample_size": None,
        "split": {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4},
        "window_size": 20,
        "window_sizes": [10, 30],
        "detector": {"kind": "isolation_forest", "n_estimators": 20, "random_state": 0},
        "evaluation": {
            "fpr_target": 0.05,
            "fixed_fpr_targets": [0.01, 0.05],
            "recall_target": 0.95,
            "val_holdout_fraction": 0.2,
        },
        "strategies": ["raw", "per_id_z", "per_id_mad"],
        "strategy": "per_id_z",
        "threshold_strategies": ["per_id_z", "global_z"],
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "phase1"),
    }


def test_residualization_group(tmp_path: Path) -> None:
    from experiments.runners.run_phase1_pird_validation import run_residualization

    cfg = _config(tmp_path)
    out = tmp_path / "phase1" / "residualization"
    df = run_residualization(cfg, out)
    assert len(df) == len(cfg["datasets"]) * len(cfg["strategies"])
    for col in ("f1", "recall", "fpr", "recall_at_fpr_0.01",
                "false_alarms_per_hour", "mean_delay_ms"):
        assert col in df.columns
    assert (out / "residualization_results.csv").exists()
    assert (out / "residualization_delta_vs_raw.csv").exists()


def test_global_threshold_group(tmp_path: Path) -> None:
    from experiments.runners.run_phase1_pird_validation import run_global_threshold

    cfg = _config(tmp_path)
    out = tmp_path / "phase1" / "global_threshold"
    df = run_global_threshold(cfg, out)
    assert len(df) == len(cfg["datasets"]) * len(cfg["threshold_strategies"]) * 2
    assert set(df["threshold_mode"]) == {"per_dataset", "global"}
    # A single pooled threshold is shared by every dataset in global mode.
    glob = df[df["threshold_mode"] == "global"]
    for _rep, g in glob.groupby("representation"):
        assert g["global_threshold"].nunique() == 1
        assert g["threshold"].nunique() == 1
    assert (out / "global_threshold_summary.csv").exists()
    assert (out / "global_thresholds.json").exists()


def test_window_sensitivity_group(tmp_path: Path) -> None:
    from experiments.runners.run_phase1_pird_validation import run_window_sensitivity

    cfg = _config(tmp_path)
    out = tmp_path / "phase1" / "window_sensitivity"
    df = run_window_sensitivity(cfg, out)
    assert len(df) == len(cfg["datasets"]) * len(cfg["window_sizes"])
    assert set(df["window_size"]) == set(cfg["window_sizes"])
    assert "ms_per_window" in df.columns
    tradeoff = pd.read_csv(out / "window_sensitivity_tradeoff.csv")
    assert len(tradeoff) == len(cfg["window_sizes"])
    assert "mean_detection_delay_ms" in tradeoff.columns


def test_cross_condition_group(tmp_path: Path) -> None:
    from experiments.runners.run_phase1_pird_validation import run_cross_condition

    cfg = _config(tmp_path)
    out = tmp_path / "phase1" / "cross_condition"
    df = run_cross_condition(cfg, out)
    assert len(df) == 2  # two ordered directions
    pairs = set(zip(df["source_condition"], df["target_condition"]))
    assert pairs == {("RPM", "gear"), ("gear", "RPM")}
    assert (df["source_condition"] != df["target_condition"]).all()
    assert "id_count_overlap" in df.columns
    assert (out / "cross_condition_results.csv").exists()


def test_run_phase1_driver_writes_summary(tmp_path: Path) -> None:
    from experiments.runners.run_phase1_pird_validation import run_phase1

    cfg = _config(tmp_path)
    result = run_phase1(cfg, mode="residualization")
    assert result["residualization_rows"] == len(cfg["datasets"]) * len(cfg["strategies"])
    assert (tmp_path / "phase1" / "run_summary.json").exists()
    assert (tmp_path / "phase1" / "config.json").exists()
