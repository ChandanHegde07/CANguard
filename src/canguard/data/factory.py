
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseDatasetLoader
from .hcrl import HCRLLoader
from .road import RoadLoader

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseDatasetLoader]] = {
    "hcrl": HCRLLoader,
    "road": RoadLoader,
}


def get_loader(name: str, data_dir: str | Path) -> BaseDatasetLoader:
    try:
        loader_cls = _REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown dataset loader '{name}'. Available: {sorted(_REGISTRY)}"
        ) from None
    logger.info("Creating %s loader", loader_cls.__name__)
    return loader_cls(data_dir)


def register_loader(name: str, loader_cls: type[BaseDatasetLoader]) -> None:
    _REGISTRY[name.lower()] = loader_cls
    logger.info("Registered loader '%s' as %s", name, loader_cls.__name__)
