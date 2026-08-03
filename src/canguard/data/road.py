"""ROAD (Real ORNL Automotive Dynamometer) CAN IDS dataset loader.

Dataset layout (treated as immutable raw data):

    <road_dir>/
        ambient/    {capture}.log + capture_metadata.json
        attacks/    {capture}.log + capture_metadata.json
        signal_extractions/   (not consumed here)
        data_table.csv
        readme.md

Raw frame format (from the ROAD readme):

    (1030000000.001020) can0 6e0#595945450000FFFF

* ``(seconds float)`` -- timestamp. Each capture uses a fixed per-capture base
  epoch, so *elapsed* time = ``timestamp - first_timestamp``.
* ``can0`` -- interface (ignored).
* ``6e0`` -- CAN ID in hex, no ``0x`` prefix.
* ``595945450000FFFF`` -- 8 payload bytes (16 hex chars). ROAD raw logs are
  uniformly DLC=8 (no variable-DLC rows).

Attack labeling:

Frames are labeled attack by matching each capture's ``injection_interval``
``[start_sec, end_sec]`` (in elapsed seconds) and ``injection_id`` from
``attacks/capture_metadata.json``:
* ``injection_id == "XXX"`` (fuzzing)  -> any frame in the interval is attack.
* ``injection_id`` is a specific AID   -> a frame is attack iff it is on that
  AID AND its timestamp lies in the injection interval.
* ``injection_id is None`` (accelerator)-> no injected message exists; those
  frames are left normal (they have no frame-level attack label).

Output: the canonical ``CAN_SCHEMA`` columns plus an extra ``attack_type``
column. ``label`` is ``"R"`` for normal and the canonical attack-type token
otherwise; ``is_attack`` is derived as ``(attack_type != "R")``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .base import CAN_SCHEMA, BaseDatasetLoader

# Matches: (123.456789) can0 HEXID#HEXDATA
_FRAME_RE = re.compile(r"^\(([0-9.]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]+)\s*$")


def _attack_type_from_name(capture_name: str) -> str:
    """Derive a canonical attack-type token from a capture filename.

    Strips the ``_masquerade`` suffix and any trailing numeric index, e.g.
    ``max_speedometer_attack_1_masquerade`` -> ``max_speedometer_attack``.
    """
    base = capture_name
    if base.endswith("_masquerade"):
        base = base[: -len("_masquerade")]
    parts = base.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts)


def _parse_log(path: Path, metadata: dict | None) -> pd.DataFrame:
    """Parse a raw ROAD ``.log`` into canonical-schema rows.

    Parameters
    ----------
    path : Path
        The ``.log`` file.
    metadata : dict | None
        Capture-level metadata entry (empty for ambient captures).

    Returns
    -------
    pd.DataFrame
        Rows with ``timestamp``, ``can_id``, ``dlc``, ``data_*``, and derived
        ``is_attack`` + ``attack_type`` columns.
    """
    metadata = metadata or {}
    path = Path(path)
    interval = metadata.get("injection_interval")
    start_sec, end_sec = (None, None)
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        start_sec, end_sec = interval
    injection_id = metadata.get("injection_id")
    attack_type = _attack_type_from_name(path.stem)

    # Vectorized parse of the raw log lines.
    lines = pd.Series(path.read_text().splitlines())
    cols = lines.str.extract(_FRAME_RE).dropna()
    if cols.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "can_id",
                "dlc",
                *[f"data_{i}" for i in range(8)],
                "is_attack",
                "attack_type",
            ]
        )
    cols = cols.reset_index(drop=True)
    ts = cols[0].astype(float)
    can_id = cols[1].str.lower()
    hex_data = cols[2].astype(str)
    first_ts = float(ts.iloc[0])
    elapsed = ts - first_ts

    # DLC / data bytes (ROAD is DLC=8; vectorized hex splits).
    dlc = (hex_data.str.len() // 2).astype(int)
    data_cols = {f"data_{i}": hex_data.str.slice(i * 2, i * 2 + 2) for i in range(8)}
    for i in range(8):
        col = data_cols[f"data_{i}"]
        data_cols[f"data_{i}"] = col.where(col.str.len() == 2, None)

    df = pd.DataFrame(
        {
            "timestamp": ts.to_numpy(),
            "can_id": can_id.to_numpy(),
            "dlc": dlc.to_numpy(),
            **{k: v.to_numpy() for k, v in data_cols.items()},
            "elapsed": elapsed.to_numpy(),
        }
    )

    in_interval = np.zeros(len(df), dtype=bool)
    if start_sec is not None and end_sec is not None:
        in_interval = (df["elapsed"].to_numpy() >= start_sec) & (
            df["elapsed"].to_numpy() <= end_sec
        )

    if injection_id == "XXX":
        is_attack = in_interval
    elif injection_id is not None:
        try:
            inj_int = int(str(injection_id), 16)
            id_vals = can_id.map(lambda x: int(str(x), 16)).to_numpy()
            is_attack = in_interval & (id_vals == inj_int)
        except ValueError:
            is_attack = np.zeros(len(df), dtype=bool)
    else:
        is_attack = np.zeros(len(df), dtype=bool)

    df["is_attack"] = is_attack.astype(int)
    df["attack_type"] = np.where(df["is_attack"] == 1, attack_type, "R")
    return df


class RoadLoader(BaseDatasetLoader):
    """Load the ROAD dataset (ambient and/or attack captures)."""

    def __init__(
        self,
        data_dir: str | Path,
        meta: dict | None = None,
        load_attacks: bool = True,
        load_ambient: bool = True,
    ) -> None:
        """Initialize with the ROAD root directory.

        Parameters
        ----------
        data_dir : str | Path
            ROAD dataset root (containing ``ambient/`` and ``attacks/``).
        meta : dict | None
            Optional merged metadata dict; if None, loaded from the capture
            metadata JSON files under ``data_dir``.
        load_attacks : bool
            Whether to include attack captures (default True).
        load_ambient : bool
            Whether to include ambient captures (default True).
        """
        super().__init__(data_dir)
        self.load_attacks = load_attacks
        self.load_ambient = load_ambient
        self._meta_all = self._load_metadata() if meta is None else meta

    def _load_metadata(self) -> dict:
        merged: dict = {}
        for key in ("attacks", "ambient"):
            meta_path = self.data_dir / key / "capture_metadata.json"
            if meta_path.exists():
                for name, entry in json.loads(meta_path.read_text()).items():
                    entry.setdefault("source", key)
                    merged[name] = entry
        return merged

    def _collect_logs(self, subdir: str) -> list[Path]:
        d = self.data_dir / subdir
        if not d.exists():
            return []
        return sorted(d.glob("*.log"))

    def load(self, sample_size: int | None = None) -> pd.DataFrame:
        """Load ROAD captures into the canonical schema.

        Parameters
        ----------
        sample_size : int | None
            If given, return only the first ``sample_size`` rows of the
            concatenated frame table.

        Returns
        -------
        pd.DataFrame
            Canonical-schema rows with an extra ``attack_type`` column and a
            derived ``is_attack`` column.
        """
        frames = []
        pairs = (("ambient", self.load_ambient), ("attacks", self.load_attacks))
        for subdir, enabled in pairs:
            if not enabled:
                continue
            for log in self._collect_logs(subdir):
                meta = _metadata_for(self._meta_all, log.stem)
                frames.append(_parse_log(log, meta))

        if not frames:
            return pd.DataFrame(columns=[*CAN_SCHEMA, "is_attack", "attack_type"])
        out = pd.concat(frames, ignore_index=True)
        out["label"] = out["attack_type"].map(lambda t: "R" if t == "R" else t)
        if sample_size is not None:
            out = out.head(int(sample_size)).copy()
        return out


def _metadata_for(meta: dict, name: str) -> dict:
    return meta.get(name, {})
