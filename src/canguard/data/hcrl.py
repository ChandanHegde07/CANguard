from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
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

    def _parse_file_head(self, path: Path, n_rows: int) -> pd.DataFrame:
        """Read only the first ``n_rows`` frames (memory-bounded equivalent of
        ``_parse_file(path).head(n_rows)``).

        HCRL rows are ragged: the payload holds exactly ``dlc`` bytes and the
        label sits at column ``3 + dlc`` (not the last column). The Python CSV
        engine pads short rows with NaN, so the label is gathered per row. This
        path is used whenever ``sample_size`` is given so that large HCRL files
        do not have to be materialised in full.
        """
        raw = pd.read_csv(
            path, header=None, names=list(range(12)), engine="python",
            nrows=int(n_rows), dtype=str,
        )
        arr = raw.to_numpy(dtype=object)
        dlc = pd.to_numeric(pd.Series(arr[:, 2]), errors="coerce").fillna(0).astype(int).to_numpy()
        idx = np.arange(len(arr))
        labels = arr[idx, np.clip(3 + dlc, 0, 11)]
        data = np.full((len(arr), 8), None, dtype=object)
        for i in range(8):
            m = i < dlc
            data[m, i] = arr[m, 3 + i]
        df = pd.DataFrame(
            {
                "timestamp": pd.to_numeric(pd.Series(arr[:, 0]), errors="coerce"),
                "can_id": pd.Series(arr[:, 1]).astype(str),
                "dlc": dlc,
                "label": pd.Series(labels).astype(str),
                **{f"data_{i}": data[:, i] for i in range(8)},
            }
        )
        df["is_attack"] = (df["label"] != "R").astype(int)
        return df

    def load(self, sample_size: int | None = None) -> pd.DataFrame:
        if sample_size is not None:
            df = self._parse_file_head(self.data_dir, int(sample_size))
        else:
            df = self._parse_file(self.data_dir)
        # Keep the canonical column order used by _parse_file.
        return df[
            ["timestamp", "can_id", "dlc", *[f"data_{i}" for i in range(8)], "label", "is_attack"]
        ]
