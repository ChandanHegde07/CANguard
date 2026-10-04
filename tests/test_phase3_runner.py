"""End-to-end smoke test for the Phase 3 transfer runner (tiny synthetic HCRL)."""

from __future__ import annotations

import csv
from pathlib import Path


def _write_tiny_hcrl_csv(path: Path, n_rows: int = 1600) -> None:
    ids = ["0316", "018f", "02b0"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for i in range(n_rows):
            cid = ids[i % 3]
            ts = 1478191030.0 + i * 0.0005
            label = "T" if (cid == "02b0" and i > n_rows - 200) else "R"
            w.writerow([f"{ts:.6f}", cid, "8", *(["01"] * 8), label])


def _cfg(tmp_path: Path) -> dict:
    return {
        "phase": 3,
        "experiment_id": "phase3_smoke",
        "seed": 0,
        "window_size": 30,
        "feature_cols": None,
        "split": {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4},
        "detector": {"kind": "isolation_forest", "n_estimators": 10, "random_state": 0},
        "evaluation": {"fpr_target": 0.01, "fixed_fpr_targets": [0.01, 0.05]},
        "source": {"dataset": "hcrl", "data_dir": str(tmp_path / "data"),
                   "datasets": ["RPM"], "sample_size": 1500},
        "target": {"dataset": "hcrl", "data_dir": str(tmp_path / "data"),
                   "datasets": ["gear"], "sample_size": 1500},
        "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "phase3" / "hcrl_to_gear"),
    }


def test_phase3_transfer_runner_smoke(tmp_path: Path) -> None:
    from experiments.runners.phase3_figures import run_figures
    from experiments.runners.run_phase3_transfer import run_dataset_transfer

    data = tmp_path / "data"
    data.mkdir()
    _write_tiny_hcrl_csv(data / "RPM_dataset.csv")
    _write_tiny_hcrl_csv(data / "gear_dataset.csv")

    cfg = _cfg(tmp_path)
    out = Path(cfg["output_dir"])
    result = run_dataset_transfer(cfg, out)

    assert "primary_f1" in result["summary"]
    assert (out / "summary.csv").exists()
    assert (out / "feature_shift.csv").exists()
    assert (out / "fixed_fpr.csv").exists()
    # Unseen handling is explicit.
    s = result["summary"]
    assert s["n_known_id_windows"] + s["n_unseen_id_windows"] == s["n_target_frames"]

    # Figures aggregate without error.
    root = tmp_path / "phase3"
    run_figures({}, root)
    assert (root / "results" / "transfer_matrix.csv").exists()
    assert (root / "plots" / "transfer_f1.png").exists()

    # Distribution-shift aggregation consumes the per-transfer tables.
    from experiments.runners.phase3_shift import run_shift

    run_shift({}, root)
    shift_dir = root / "distribution_shift"
    assert (shift_dir / "all_feature_shift.csv").exists()
    assert (shift_dir / "shift_summary.csv").exists()


def test_hcrl_window_table_uses_section_not_source(tmp_path: Path) -> None:
    """Regression: target HCRL windows must follow the *target* section config."""
    from canguard.exp import FeatureCache
    from experiments.runners.run_phase3_transfer import _hcrl_window_table

    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    _write_tiny_hcrl_csv(d1 / "RPM_dataset.csv", n_rows=400)
    _write_tiny_hcrl_csv(d2 / "RPM_dataset.csv", n_rows=1200)

    cache = FeatureCache(tmp_path / "cache")
    cfg = {"window_size": 30}
    w1 = _hcrl_window_table("RPM", {"data_dir": str(d1)}, cfg, cache)
    w2 = _hcrl_window_table("RPM", {"data_dir": str(d2)}, cfg, cache)
    assert len(w2) > len(w1)

    # A section-level sample_size must cap the loaded frames.
    w3 = _hcrl_window_table("RPM", {"data_dir": str(d2), "sample_size": 200}, cfg, cache)
    assert len(w3) < len(w2)


def test_phase3_unknown_dataset_raises_clear_error(tmp_path: Path) -> None:
    import pytest

    from experiments.runners.run_phase3_transfer import run_dataset_transfer

    cfg = _cfg(tmp_path)
    cfg["target"] = {"dataset": "not_a_real_dataset", "data_dir": str(tmp_path / "x")}
    with pytest.raises(ValueError, match="Unsupported target dataset"):
        run_dataset_transfer(cfg, tmp_path / "phase3" / "unknown")


def test_phase3_missing_gem_data_raises_clear_error(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from canguard.data.gem_can import resolve_gem_root

    monkeypatch.chdir(tmp_path)  # neutralise repo-relative fallbacks
    with pytest.raises(FileNotFoundError, match="GEM-CAN"):
        resolve_gem_root(tmp_path / "absent_gem")


def test_phase3_vehicle_transfer_missing_data_raises_clear_error(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    from canguard.data.survival import resolve_survival_root

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Survival"):
        resolve_survival_root(tmp_path / "absent_survival")
