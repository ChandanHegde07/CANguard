"""Phase 3 tests for GEM-CAN and HCRL Survival (loaders + transfer paths).

All fixtures are tiny synthetic files written in the *real* on-disk schemas so
the tests exercise the parsing logic without depending on the large datasets.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from canguard.data.gem_can import GemCanLoader
from canguard.data.survival import SurvivalLoader


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _write_gem(path: Path, n_rows: int, attack: bool) -> None:
    ids = ["0316", "018F", "18FF00F9"]
    header = ["timestamp", "arbitration_id", "dlc", *[f"data{i}" for i in range(8)]]
    if attack:
        header += ["label", "Attack_Type"]
    else:
        header += ["label"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n_rows):
            cid = ids[i % len(ids)]
            ts = 1000.0 + i * 0.001
            data = [f"{i % 256:02X}"] * 8
            if attack:
                is_atk = 1 if cid == "18FF00F9" and i > n_rows // 2 else 0
                atype = "DoS" if is_atk else "Normal"
                w.writerow([f"{ts:.3f}", cid, 8, *data, is_atk, atype])
            else:
                w.writerow([f"{ts:.3f}", cid, 8, *data, 0])


def _write_gem_tree(root: Path, n_rows: int = 900) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_gem(root / "GEM_Normal_Driving_Raw_Labeled.csv", n_rows, attack=False)
    _write_gem(root / "GEM_Attack_Scenario_Raw_Labeled.csv", n_rows, attack=True)
    return root


def _write_survival_file(path: Path, n_rows: int, ids, labeled: bool,
                         bundle: bool, attack_ids=()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        for i in range(n_rows):
            cid = ids[i % len(ids)]
            ts = 1_500_000_000.0 + i * 0.001
            data = [f"{i % 256:02X}"] * 8
            is_atk = cid in attack_ids and i > n_rows // 2
            if not labeled:
                if bundle:
                    f.write(f"{ts:.6f},{cid},8,{' '.join(data)}\n")
                else:
                    f.write(f"{ts:.6f},{cid},8,{','.join(data)}\n")
            else:
                lab = "T" if is_atk else "R"
                f.write(f"{ts:.6f},{cid},8,{','.join(data)},{lab}\n")


def _write_survival_tree(root: Path, n_rows: int = 500) -> Path:
    base = root / "dataset"
    ids = ["0316", "018F", "02B0"]
    _write_survival_file(base / "Sonata" / "FreeDrivingData_SONATA.txt",
                         n_rows, ids, labeled=False, bundle=False)
    _write_survival_file(base / "Sonata" / "Flooding_dataset_SONATA.txt",
                         n_rows, ids, labeled=True, bundle=False, attack_ids=("02B0",))
    _write_survival_file(base / "Soul" / "FreeDrivingData_KIA.txt",
                         n_rows, ids, labeled=False, bundle=True)
    _write_survival_file(base / "Soul" / "Fuzzy_dataset_KIA.txt",
                         n_rows, ids, labeled=True, bundle=False, attack_ids=("018F",))
    _write_survival_file(base / "Spark" / "FreeDrivingData_Spark.txt",
                         n_rows, ids, labeled=False, bundle=True)
    _write_survival_file(base / "Spark" / "Malfunction18E_dataset_Spark.txt",
                         n_rows, ids, labeled=True, bundle=False, attack_ids=("0316",))
    return root


# ---------------------------------------------------------------------------
# GEM-CAN loader
# ---------------------------------------------------------------------------
def test_gem_loader_schema_and_labels(tmp_path: Path) -> None:
    root = _write_gem_tree(tmp_path / "gem")
    loader = GemCanLoader(root)
    assert loader.list_captures() == ["normal_driving", "attack_scenario"]

    norm = loader.load_capture("normal_driving")
    assert list(norm.columns) == [
        "timestamp", "can_id", "dlc", "data_0", "data_1", "data_2", "data_3",
        "data_4", "data_5", "data_6", "data_7", "label",
        "is_attack", "attack_type", "capture",
    ]
    assert norm["is_attack"].sum() == 0
    assert (norm["attack_type"] == "Normal").all()
    # 29-bit extended ID lower-cased, timestamp numeric.
    assert "18ff00f9" in set(norm["can_id"])
    assert pd.api.types.is_float_dtype(norm["timestamp"])

    atk = loader.load_capture("attack_scenario")
    assert atk["is_attack"].sum() > 0
    assert set(atk.loc[atk["is_attack"] == 1, "attack_type"]) == {"DoS"}
    assert set(atk.loc[atk["is_attack"] == 0, "attack_type"]) == {"Normal"}
    # label and is_attack agree
    assert ((atk["label"] == "T") == (atk["is_attack"] == 1)).all()


def test_gem_loader_root_resolution(tmp_path: Path) -> None:
    root = _write_gem_tree(tmp_path / "GEM_CAN_Dataset_R2" / "GEM_CAN_Dataset_R2")
    # Passing either the parent or the leaf resolves to the same files.
    assert GemCanLoader(tmp_path / "GEM_CAN_Dataset_R2").data_dir == root.resolve()
    assert GemCanLoader(root).data_dir == root.resolve()


# ---------------------------------------------------------------------------
# Survival loader
# ---------------------------------------------------------------------------
def test_survival_loader_vehicle_discovery(tmp_path: Path) -> None:
    root = _write_survival_tree(tmp_path / "survival")
    loader = SurvivalLoader(root)
    assert loader.list_vehicles() == ["chevrolet_spark", "kia_soul", "sonata"]
    caps = loader.list_captures("sonata")
    types = {atype for _, _, atype in caps}
    assert types == {"normal", "flooding"}
    # Display mapping is explicit and stable.
    from canguard.data.survival import VEHICLE_DISPLAY
    assert VEHICLE_DISPLAY["kia_soul"] == "Kia Soul"
    assert VEHICLE_DISPLAY["chevrolet_spark"] == "Chevrolet Spark"


def test_survival_loader_parses_both_payload_formats(tmp_path: Path) -> None:
    root = _write_survival_tree(tmp_path / "survival")
    loader = SurvivalLoader(root)
    # Sonata free-driving: comma-separated payload, no label column.
    sonata = loader.load_capture("sonata", "FreeDrivingData_SONATA")
    assert sonata["is_attack"].sum() == 0
    assert sonata["data_0"].notna().all()
    # KIA free-driving: space-separated payload bundle, no label column.
    kia = loader.load_capture("kia_soul", "FreeDrivingData_KIA")
    assert kia["is_attack"].sum() == 0
    assert kia["data_7"].notna().all()
    # Labeled attack capture.
    flood = loader.load_capture("sonata", "Flooding_dataset_SONATA")
    assert flood["is_attack"].sum() > 0
    assert flood.loc[flood["is_attack"] == 1, "attack_type"].eq("flooding").all()
    assert flood["vehicle"].eq("sonata").all()


def test_survival_vehicle_metadata_isolated(tmp_path: Path) -> None:
    root = _write_survival_tree(tmp_path / "survival")
    loader = SurvivalLoader(root)
    for veh in loader.list_vehicles():
        df = loader.load_vehicle(veh)
        assert set(df["vehicle"]) == {veh}
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Transfer integration (tiny end-to-end, zero-shot protocol)
# ---------------------------------------------------------------------------
def _tiny_hcrl(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    ids = ["0316", "018F", "02B0"]
    with open(data_dir / "RPM_dataset.csv", "w", newline="") as f:
        w = csv.writer(f)
        for i in range(2400):
            cid = ids[i % 3]
            ts = 1_478_000_000.0 + i * 0.0005
            label = "T" if (cid == "02B0" and i > 2000) else "R"
            w.writerow([f"{ts:.6f}", cid, "8", *(["01"] * 8), label])


def test_gem_dataset_transfer_dispatch_and_isolation(tmp_path: Path) -> None:
    from canguard.exp import FeatureCache
    from experiments.runners.run_phase3_transfer import _build_source, _build_target

    data = tmp_path / "data"
    _tiny_hcrl(data)
    gem = _write_gem_tree(tmp_path / "gem")
    cfg = {
        "seed": 0, "window_size": 30, "feature_cols": None,
        "split": {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4},
        "detector": {"kind": "isolation_forest", "n_estimators": 10, "random_state": 0},
        "source": {"dataset": "hcrl", "data_dir": str(data),
                   "datasets": ["RPM"], "sample_size": 2400},
        "target": {"dataset": "gem_can", "data_dir": str(gem)},
    }
    cache = FeatureCache(tmp_path / "cache")
    calib, ifv, val = _build_source(cfg, cache)
    assert len(calib) > 0 and len(ifv) > 0 and len(val) > 0
    # Source normals only.
    assert (calib["is_attack"] == 0).all() and (ifv["is_attack"] == 0).all()
    target = _build_target(cfg, cache)
    assert not target.empty
    # Target provenance is the GEM captures.
    assert set(target["capture"]) <= {"normal_driving", "attack_scenario"}


def test_gem_source_to_hcrl_dispatch(tmp_path: Path) -> None:
    from canguard.exp import FeatureCache
    from experiments.runners.run_phase3_transfer import _build_source, _build_target

    data = tmp_path / "data"
    _tiny_hcrl(data)
    gem = _write_gem_tree(tmp_path / "gem")
    cfg = {
        "seed": 0, "window_size": 30, "feature_cols": None,
        "split": {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4},
        "source": {"dataset": "gem_can", "data_dir": str(gem)},
        "target": {"dataset": "hcrl", "data_dir": str(data), "datasets": ["RPM"],
                   "sample_size": 2400},
    }
    cache = FeatureCache(tmp_path / "cache")
    calib, ifv, val = _build_source(cfg, cache)
    assert (calib["is_attack"] == 0).all() and (ifv["is_attack"] == 0).all()
    target = _build_target(cfg, cache)
    assert len(target) > 0
    assert set(target["can_id"]) <= {"316", "18f", "2b0"}


def test_vehicle_transfer_smoke(tmp_path: Path) -> None:
    from experiments.runners.run_phase3_transfer import run_vehicle_transfer

    root = _write_survival_tree(tmp_path / "survival")
    cfg = {
        "seed": 0, "window_size": 10, "feature_cols": None,
        "detector": {"kind": "isolation_forest", "n_estimators": 10, "random_state": 0},
        "evaluation": {"fpr_target": 0.05, "fixed_fpr_targets": [0.05]},
        "dataset": {"data_dir": str(root), "vehicles": ["sonata", "kia_soul", "chevrolet_spark"]},
        "run_recalibration_diagnostic": True,
        "cache_dir": str(tmp_path / "cache"),
    }
    out = tmp_path / "survival_vehicle"
    res = run_vehicle_transfer(cfg, out)
    vt = pd.read_csv(out / "vehicle_transfer.csv")
    assert len(vt) == 6  # 3 vehicles -> 6 ordered pairs
    # No self-transfer and source/target vehicles always differ.
    assert (vt["source_vehicle"] != vt["target_vehicle"]).all()
    assert set(vt["source_vehicle"]) == {"sonata", "kia_soul", "chevrolet_spark"}
    assert (out / "recalibration_summary.csv").exists()
    # The threshold depends only on the source vehicle, never on the target.
    for src, g in vt.groupby("source_vehicle"):
        assert g["source_threshold"].nunique() == 1
    assert "vehicle_transfer" in res


def test_gem_capture_transfer_smoke(tmp_path: Path) -> None:
    from experiments.runners.run_phase3_transfer import run_gem_capture_transfer

    gem = _write_gem_tree(tmp_path / "gem")
    cfg = {
        "seed": 0, "window_size": 10, "feature_cols": None,
        "detector": {"kind": "isolation_forest", "n_estimators": 10, "random_state": 0},
        "evaluation": {"fpr_target": 0.05, "fixed_fpr_targets": [0.05]},
        "dataset": {"data_dir": str(gem)},
        "cache_dir": str(tmp_path / "cache"),
    }
    out = tmp_path / "gem_captures"
    run_gem_capture_transfer(cfg, out)
    df = pd.read_csv(out / "capture_transfer.csv")
    # Two independent captures -> two ordered directions.
    assert len(df) == 2
    assert set(df["source_capture"]) == {"normal_driving", "attack_scenario"}
    assert (df["source_capture"] != df["target_capture"]).all()
