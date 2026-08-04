
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .base import BaseDatasetLoader


class HCRLLoader(BaseDatasetLoader):
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
        df = self._parse_file(self.data_dir)
        if sample_size is not None:
            df = df.head(int(sample_size)).copy()
        return df
