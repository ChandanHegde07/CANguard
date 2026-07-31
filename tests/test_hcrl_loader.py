"""Verify HCRLLoader matches the notebook's inline loader byte-for-byte."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from canguard.data import HCRLLoader, get_loader


def _notebook_loader(path: Path) -> pd.DataFrame:
    """Reference implementation copied verbatim from eda_hcrl.ipynb."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            timestamp = row[0]
            can_id = row[1]
            dlc = int(row[2])
            data = {f"data_{i}": row[3 + i] if i < dlc else None for i in range(8)}
            label = row[3 + dlc]
            rows.append(
                {
                    "timestamp": timestamp,
                    "can_id": can_id,
                    "dlc": dlc,
                    **data,
                    "label": label,
                }
            )
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].astype(float)
    df["dlc"] = df["dlc"].astype(int)
    df["is_attack"] = (df["label"] != "R").astype(int)
    assert df["label"].isna().sum() == 0
    return df


def _write_tiny_csv(tmp_path: Path) -> Path:
    p = tmp_path / "Tiny.csv"
    lines = [
        "1478191030.045114,0316,8,05,22,68,09,22,20,00,75,R",
        "1478191030.045353,018f,8,fe,3b,00,00,00,3c,00,00,R",
        # DLC=5 row (label at index 3+5=8) — must pad data_5..7
        "1478195721.905736,02b0,5,ff,7f,00,05,49,R",
        "1478195721.908437,0002,8,00,00,00,00,00,01,07,15,R",
        # An attack row (T)
        "1478195722.000000,0000,8,aa,bb,cc,dd,ee,ff,00,11,T",
    ]
    p.write_text("\n".join(lines) + "\n")
    return p


def test_hcrl_matches_notebook_loader_schema_and_values(tmp_path: Path) -> None:
    csv_path = _write_tiny_csv(tmp_path)

    ref = _notebook_loader(csv_path)
    ldr = HCRLLoader(csv_path)
    got = ldr.load(sample_size=None)

    pd.testing.assert_frame_equal(got, ref)


def test_hcrl_sample_size_head_behavior(tmp_path: Path) -> None:
    csv_path = _write_tiny_csv(tmp_path)
    ref = _notebook_loader(csv_path).head(3)
    got_sample = HCRLLoader(csv_path).load(sample_size=3)
    assert len(got_sample) == 3
    assert list(got_sample["label"]) == list(ref["label"])


def test_hcrl_variable_dlc_padding(tmp_path: Path) -> None:
    csv_path = _write_tiny_csv(tmp_path)
    df = HCRLLoader(csv_path).load()
    dlc5 = df[df["dlc"] == 5].iloc[0]
    assert dlc5["data_0"] == "ff"
    assert dlc5["data_4"] == "49"
    # missing bytes are NaN
    assert pd.isna(dlc5["data_5"])
    assert pd.isna(dlc5["data_7"])
    # label is not corrupted
    assert dlc5["label"] == "R"


def test_hcrl_is_attack_flag(tmp_path: Path) -> None:
    csv_path = _write_tiny_csv(tmp_path)
    df = HCRLLoader(csv_path).load()
    assert (df["is_attack"] == (df["label"] != "R")).all()
    assert df["is_attack"].sum() == 1  # only the T row


def test_hcrl_no_missing_labels(tmp_path: Path) -> None:
    csv_path = _write_tiny_csv(tmp_path)
    df = HCRLLoader(csv_path).load()
    assert df["label"].isna().sum() == 0


def test_get_loader_factory_registry(tmp_path: Path) -> None:
    csv_path = _write_tiny_csv(tmp_path)
    ldr = get_loader("hcrl", csv_path)
    assert isinstance(ldr, HCRLLoader)
    assert ldr.load().shape == get_loader("hcrl", csv_path).load().shape


def test_factory_unknown_loader_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataset loader"):
        get_loader("nope", tmp_path)
