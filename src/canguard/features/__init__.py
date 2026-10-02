"""Feature extraction for CAN windows."""

from .global_features import (
    ALL_GLOBAL_FEATURES,
    BUS_FEATURES,
    CROSS_ID_FEATURES,
    FEATURE_CAUSALITY,
    GLOBAL_ABLATIONS,
    GLOBAL_FEATURE_FAMILIES,
    ID_POPULATION_FEATURES,
    GlobalFeatureConfig,
    GlobalStats,
    build_global_features,
    fit_global_stats,
    transform_global,
)
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
    "ALL_GLOBAL_FEATURES",
    "BEHAVIORAL_FEATURES",
    "BEHAVIORAL_FEATURES_V1",
    "BUS_FEATURES",
    "CROSS_ID_FEATURES",
    "EPS",
    "FEATURE_CAUSALITY",
    "FeaturePipeline",
    "GLOBAL_ABLATIONS",
    "GLOBAL_FEATURE_FAMILIES",
    "GROUP_DLC",
    "GROUP_FLAT_BYTE",
    "GROUP_IAT",
    "GROUP_OTHER",
    "GlobalCANContext",
    "GlobalFeatureConfig",
    "GlobalStats",
    "ID_POPULATION_FEATURES",
    "MIN_WINDOWS_PER_ID",
    "PerIDWindow",
    "build_global_features",
    "fit_global_stats",
    "fit_known_ids_on_normal_prefix",
    "fit_per_id_stats",
    "temporal_split",
    "transform_global",
    "transform_residuals",
]
