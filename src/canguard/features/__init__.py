"""Feature extraction for CAN windows."""

from .groups import (
    BEHAVIORAL_FEATURES,
    BEHAVIORAL_FEATURES_V1,
    GROUP_DLC,
    GROUP_FLAT_BYTE,
    GROUP_IAT,
    GROUP_OTHER,
)
from .per_id import (
    EPS,
    MIN_WINDOWS_PER_ID,
    fit_per_id_stats,
    transform_residuals,
)
from .splits import temporal_split
from .window import (
    FeaturePipeline,
    GlobalCANContext,
    PerIDWindow,
    fit_known_ids_on_normal_prefix,
)

__all__ = [
    "BEHAVIORAL_FEATURES",
    "BEHAVIORAL_FEATURES_V1",
    "EPS",
    "FeaturePipeline",
    "GlobalCANContext",
    "GROUP_DLC",
    "GROUP_FLAT_BYTE",
    "GROUP_IAT",
    "GROUP_OTHER",
    "MIN_WINDOWS_PER_ID",
    "PerIDWindow",
    "fit_known_ids_on_normal_prefix",
    "fit_per_id_stats",
    "temporal_split",
    "transform_residuals",
]
