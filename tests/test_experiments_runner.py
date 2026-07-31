"""Smoke test: the runner executes a config end-to-end on a small synthetic case.

We avoid the real HCRL CSVs by pointing the config at a tiny generated CSV so
the runner paths (loader -> features -> residuals -> detector -> metrics) all
resolve without a large dataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_tiny_hcrl_csv(path: Path, n_rows: int = 1600) -> None:
    rows = []
    # Heavily normal first (so train normals are plentiful), occasional attack.
    for i in range(n_rows):
        cid = ["0316", "018f", "02b0"][i % 3]
        ts = 1478191030.0 + i * 0.0005
        data = ["01"] * 8
        label = "T" if i > n_rows - 150 and i % 15 == 0 else "R"
        rows.append([f"{ts:.6f}", cid, "8", *data, label])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_train_detector_runner_smoke(tmp_path: Path) -> None:
    from experiments.runners.train_detector import run_experiment

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_tiny_hcrl_csv(data_dir / "RPM_dataset.csv")
    _write_tiny_hcrl_csv(data_dir / "gear_dataset.csv")

    config = {
        "dataset": "hcrl",
        "data_dir": str(data_dir),
        "sample_size": 1500,
        "window_size": 30,
        "datasets": ["RPM", "gear"],
        "split": {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4},
        "detector": {"kind": "isolation_forest", "n_estimators": 30, "random_state": 0},
        "evaluation": {"val_holdout_fraction": 0.2, "fpr_target": 0.01},
        "output_dir": str(tmp_path / "results"),
    }
    results = run_experiment(config)
    assert set(results.keys()) == {"RPM", "gear"}
    for name in results:
        assert "f1" in results[name]
        assert "recall" in results[name]
    # JSON serializable
    json.dumps(results)


def test_ablation_runner_feature_resolution() -> None:
    from experiments.runners.train_detector import _resolve_feature_cols

    assert _resolve_feature_cols({"variant_feature_groups": ["iat"]}) == [
        "iat_mean",
        "iat_std",
        "iat_median",
        "iat_min",
        "iat_max",
    ]
    # With no variant, falls back to full V1 set.
    full = _resolve_feature_cols({})
    from canguard.features import BEHAVIORAL_FEATURES_V1

    assert full == list(BEHAVIORAL_FEATURES_V1)
