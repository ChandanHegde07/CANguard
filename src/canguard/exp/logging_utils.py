"""Logging helpers for experiment runners."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger for console (+ optional file)."""
    logger = logging.getLogger("canguard")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
