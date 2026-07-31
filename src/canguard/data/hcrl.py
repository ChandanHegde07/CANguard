"""HCRL Car-Hacking Dataset loader.

DLC-aware parser: rows with ``dlc < 8`` contain fewer data fields on disk.
This parser reads exactly ``dlc`` data bytes and pads ``data_*`` columns for
missing positions with ``None`` (rendered as NaN by pandas). The label is at
column index ``3 + dlc``, so it is never corrupted by positional misalignment.

This is a logic-preserving port of the inline loader in ``eda_hcrl.ipynb``.
``data_dir`` is interpreted as the path to a single HCRL CSV file, matching
the notebook's ``load_hcrl(path)`` call signature exactly.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .base import BaseDatasetLoader


class HCRLLoader(BaseDatasetLoader):
    """Load the HCRL Car-Hacking dataset (DoS, Fuzzy, RPM, Gear)."""

    def _parse_file(self, path: Path) -> pd.DataFrame:
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

    def load(self, sample_size: int | None = None) -> pd.DataFrame:
        """Load the HCRL dataset.

        Parameters
        ----------
        sample_size : int | None
            If given, return only the first ``sample_size`` rows.

        Returns
        -------
        pd.DataFrame
            A DataFrame in the canonical schema, with an added ``is_attack``
            column (int: 1 if attack, 0 if normal).
        """
        df = self._parse_file(self.data_dir)
        if sample_size is not None:
            df = df.head(int(sample_size)).copy()
        return df
