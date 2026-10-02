"""Attack generation for Phase 2 (white-box adaptive / gradual drift)."""

from .adaptive import (
    DEFAULT_ATTACKABLE_FEATURES,
    DRIFT_RATE_FRACTION,
    FEATURE_BOUNDS,
    INTEGER_FEATURES,
    NON_ATTACKABLE,
    VALID_ATTACK_MODES,
    VALID_DIRECTIONS,
    VALID_DRIFT_RATES,
    VALID_DRIFT_SHAPES,
    AttackParameters,
    apply_alpha_series,
    apply_fixed_attack,
    attack_feature_matrix,
    gradual_alphas,
    gradual_attack,
    resolve_target_features,
)

__all__ = [
    "AttackParameters",
    "DEFAULT_ATTACKABLE_FEATURES",
    "DRIFT_RATE_FRACTION",
    "FEATURE_BOUNDS",
    "INTEGER_FEATURES",
    "NON_ATTACKABLE",
    "VALID_ATTACK_MODES",
    "VALID_DIRECTIONS",
    "VALID_DRIFT_RATES",
    "VALID_DRIFT_SHAPES",
    "apply_alpha_series",
    "apply_fixed_attack",
    "attack_feature_matrix",
    "gradual_alphas",
    "gradual_attack",
    "resolve_target_features",
]
