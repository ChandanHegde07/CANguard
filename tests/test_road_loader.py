"""Tests for the ROAD dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

from canguard.data import RoadLoader, get_loader


def _write_metadata(root: Path, entries: dict) -> None:
    (root / "attacks").mkdir(parents=True, exist_ok=True)
    (root / "ambient").mkdir(parents=True, exist_ok=True)
    (root / "attacks" / "capture_metadata.json").write_text(json.dumps(entries))
    (root / "ambient" / "capture_metadata.json").write_text(json.dumps({}))


def _write_log(root: Path, kind: str, name: str, lines: list[str]) -> None:
    (root / kind / f"{name}.log").write_text("\n".join(lines) + "\n")


def _ambient_lines(n=50, base=1030000000.0):
    return [f"({base + i * 0.01:.6f}) can0 0d0#{i:016X}" for i in range(n)]


def _frame(id_hex, timestamp, data="0011223344556677", iface="can0"):
    return f"({timestamp:.6f}) {iface} {id_hex}#{data}"


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "road"
    _write_metadata(
        root,
        {
            "max_speedometer_attack_1": {
                "elapsed_sec": 20.0,
                "injection_data_str": "XXXXXXXXXXFFXXXX",
                "injection_id": "0xd0",
                "injection_interval": [5.0, 15.0],
                "on_dyno": True,
            },
            "fuzzing_attack_1": {
                "elapsed_sec": 10.0,
                "injection_data_str": "FFFFFFFFFFFFFFFF",
                "injection_id": "XXX",
                "injection_interval": [3.0, 8.0],
                "on_dyno": True,
            },
        },
    )
    return root


def test_parse_meta_injected_id_matches_zero_padded_log_hex(tmp_path: Path):
    root = _make_root(tmp_path)
    base = 1030000000.0
    # ROAD logs are timestamp-ascending; first frame (elapsed 0) sets the clock.
    lines = [
        _frame("0D0", base + 0.0),  # elapsed 0 -> normal (first_ts)
        _frame("0D0", base + 6.0),  # elapsed 6 -> in [5,15], id 0xd0 -> attack
        _frame("033", base + 6.0),  # different id, in interval -> normal
    ]
    _write_log(root, "attacks", "max_speedometer_attack_1", lines)
    meta = json.loads((root / "attacks" / "capture_metadata.json").read_text())

    from canguard.data.road import _parse_log

    df = _parse_log(
        root / "attacks" / "max_speedometer_attack_1.log", meta["max_speedometer_attack_1"]
    )
    assert int(df["is_attack"].sum()) == 1
    assert df.iloc[1]["is_attack"] == 1


def test_parse_fuzzing_all_ids_in_interval(tmp_path: Path):
    root = _make_root(tmp_path)
    base = 1030000000.0
    lines = [
        _frame("0D0", base + 0.0),  # elapsed 0 -> normal (first_ts)
        _frame("0D0", base + 4.0),  # elapsed 4 -> [3,8] -> attack
        _frame("033", base + 6.0),  # elapsed 6 -> [3,8] -> attack (fuzz hits all)
    ]
    _write_log(root, "attacks", "fuzzing_attack_1", lines)
    meta = json.loads((root / "attacks" / "capture_metadata.json").read_text())
    from canguard.data.road import _parse_log

    df = _parse_log(str(root / "attacks" / "fuzzing_attack_1.log"), meta["fuzzing_attack_1"])
    assert int(df["is_attack"].sum()) == 2


def test_ambient_all_normal(tmp_path: Path):
    root = _make_root(tmp_path)
    _write_log(root, "ambient", "ambient_dyno_idle", _ambient_lines())
    from canguard.data.road import _parse_log

    df = _parse_log(str(root / "ambient" / "ambient_dyno_idle.log"), None)
    assert df["is_attack"].sum() == 0
    assert df["attack_type"].eq("R").all()


def test_load_concat_attack_and_ambient(tmp_path: Path):
    root = _make_root(tmp_path)
    _write_log(root, "ambient", "ambient_dyno_idle", _ambient_lines())
    attack_lines = [
        _frame("0D0", 1030000000.0 + 0.0),  # elapsed 0 (first_ts)
        _frame("0D0", 1030000000.0 + 10.0),  # elapsed 10 -> in [5,15] -> attack
    ]
    _write_log(root, "attacks", "max_speedometer_attack_1", attack_lines)

    ldr = RoadLoader(root)
    df = ldr.load()
    # canonical schema present
    for col in (
        "timestamp",
        "can_id",
        "dlc",
        "data_0",
        "data_1",
        "data_2",
        "data_3",
        "data_4",
        "data_5",
        "data_6",
        "data_7",
        "label",
        "attack_type",
        "is_attack",
    ):
        assert col in df.columns
    assert int(df["is_attack"].sum()) == 1
    assert "attack_type" in df.columns
    assert set(df["attack_type"].unique()) <= {"R", "max_speedometer_attack"}


def test_load_sample_size(tmp_path: Path):
    root = _make_root(tmp_path)
    _write_log(root, "ambient", "ambient_dyno_idle", _ambient_lines(n=100))
    df = RoadLoader(root).load(sample_size=50)
    assert len(df) == 50


def test_factory_registration_road(tmp_path: Path):
    root = _make_root(tmp_path)
    _write_log(root, "ambient", "ambient_dyno_idle", _ambient_lines())
    ldr = get_loader("road", root)
    assert isinstance(ldr, RoadLoader)
