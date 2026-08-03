"""Disk cache for expensive feature / residual tables."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("canguard")


class FeatureCache:
    """Simple parquet/pickle cache keyed by a JSON-serializable fingerprint."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, fingerprint: dict[str, Any]) -> str:
        payload = json.dumps(fingerprint, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def path_for(self, fingerprint: dict[str, Any], suffix: str = ".parquet") -> Path:
        return self.root / f"{self._key(fingerprint)}{suffix}"

    def load_df(self, fingerprint: dict[str, Any]) -> pd.DataFrame | None:
        path = self.path_for(fingerprint)
        if not path.exists():
            return None
        logger.info("cache hit: %s", path.name)
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)

    def save_df(self, fingerprint: dict[str, Any], df: pd.DataFrame) -> Path:
        # Prefer parquet; fall back to pickle if engine missing.
        path = self.path_for(fingerprint, ".parquet")
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = self.path_for(fingerprint, ".pkl")
            df.to_pickle(path)
        meta = path.with_suffix(path.suffix + ".meta.json")
        meta.write_text(json.dumps(fingerprint, indent=2, default=str), encoding="utf-8")
        logger.info("cache write: %s", path.name)
        return path
