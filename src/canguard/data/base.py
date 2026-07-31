"""Abstract dataset loader contract.

Every loader must produce a DataFrame in the canonical CAN guard schema so
that the feature pipeline, detectors, and evaluation code are dataset-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

# Canonical ordered columns produced by every loader.
CAN_SCHEMA = [
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
]


class BaseDatasetLoader(ABC):
    """Load a CAN trace dataset into the canonical schema.

    Subclasses implement :meth:`load`. The canonical schema guarantees
    ``timestamp`` (float, seconds), ``can_id`` (str, hex, lower-case, no
    ``0x`` prefix as-is), ``dlc`` (int), ``data_*`` (str hex bytes, NaN-padded
    when ``dlc < 8``), and ``label`` (str; ``"R"`` means normal on HCRL,
    other values mark attack classes).
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Store the data path containing the dataset file(s)."""
        self.data_dir = Path(data_dir)

    @abstractmethod
    def load(self, sample_size: int | None = None) -> pd.DataFrame:
        """Return a DataFrame matching :data:`CAN_SCHEMA`.

        Parameters
        ----------
        sample_size : int | None
            If given, return only the first ``sample_size`` rows. If ``None``,
            return the entire dataset.
        """
        raise NotImplementedError

    def requires_download(self) -> bool:
        """Return True if the data must be downloaded before loading."""
        return False
