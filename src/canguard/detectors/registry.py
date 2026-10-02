
from __future__ import annotations

from typing import Any, Callable

from .autoencoder import AutoencoderDetector
from .base import BaseAnomalyDetector
from .elliptic_envelope import EllipticEnvelopeDetector
from .hbos import HBOSDetector
from .isolation_forest import IsolationForestDetector
from .lof import LOFDetector
from .one_class_svm import OneClassSVMDetector
from .sequence_autoencoder import SequenceAutoencoderDetector

DetectorFactory = Callable[..., BaseAnomalyDetector]

_REGISTRY: dict[str, DetectorFactory] = {
    "isolation_forest": IsolationForestDetector,
    "one_class_svm": OneClassSVMDetector,
    "lof": LOFDetector,
    "hbos": HBOSDetector,
    "elliptic_envelope": EllipticEnvelopeDetector,
    "autoencoder": AutoencoderDetector,
    # External comparison baseline (CANet / sequence-AE inspired); not a SOTA claim.
    "sequence_autoencoder": SequenceAutoencoderDetector,
}


def list_detectors() -> list[str]:
    return sorted(_REGISTRY)


def create_detector(kind: str, **kwargs: Any) -> BaseAnomalyDetector:
    key = kind.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown detector '{kind}'. Available: {list_detectors()}")
    # Filter kwargs to those accepted by the constructor where possible.
    cls = _REGISTRY[key]
    # Common seed passthrough.
    return cls(**{k: v for k, v in kwargs.items() if _accepts(cls, k)})


def _accepts(cls: type, name: str) -> bool:
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return True
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return name in sig.parameters


def register_detector(name: str, factory: DetectorFactory) -> None:
    _REGISTRY[name.lower()] = factory
