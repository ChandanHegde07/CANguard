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

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    @abstractmethod
    def load(self, sample_size: int | None = None) -> pd.DataFrame:
       
        raise NotImplementedError

    def requires_download(self) -> bool:
        return False
