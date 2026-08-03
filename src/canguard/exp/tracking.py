"""Experiment run directories and result persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("canguard")


class ExperimentRun:
    """Creates a timestamped run directory under an experiment root."""

    def __init__(self, root: str | Path, name: str = "phase_a") -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.root = Path(root)
        self.run_dir = self.root / name / "runs" / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("canguard")

    def path(self, *parts: str) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_config(self, config: dict[str, Any]) -> Path:
        return save_json(self.path("config.json"), config)


def save_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def save_csv(path: str | Path, df: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("wrote %s", path)
    return path
