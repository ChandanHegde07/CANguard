"""Factory for dataset loaders."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseDatasetLoader
from .hcrl import HCRLLoader

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseDatasetLoader]] = {
    "hcrl": HCRLLoader,
}


def get_loader(name: str, data_dir: str | Path) -> BaseDatasetLoader:
    """Return a configured loader instance for ``name``.

    Parameters
    ----------
    name : str
        Dataset name key (e.g. ``"hcrl"``, later ``"road"``).
    data_dir : str | Path
        Path the loader reads from (per-loader semantics; see each loader).
    """
    try:
        loader_cls = _REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown dataset loader '{name}'. Available: {sorted(_REGISTRY)}"
        ) from None
    logger.info("Creating %s loader", loader_cls.__name__)
    return loader_cls(data_dir)


def register_loader(name: str, loader_cls: type[BaseDatasetLoader]) -> None:
    """Register a new loader (used to add ROAD without touching core)."""
    _REGISTRY[name.lower()] = loader_cls
    logger.info("Registered loader '%s' as %s", name, loader_cls.__name__)
