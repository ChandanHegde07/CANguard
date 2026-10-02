
from __future__ import annotations

# v1 behavioral feature set (flat, per-ID window). Verbatim from
# feature_eng_hcrl.ipynb / pird_hcrl.ipynb.
BEHAVIORAL_FEATURES_V1: list[str] = [
    "iat_mean",
    "iat_std",
    "iat_median",
    "iat_min",
    "iat_max",
    "dlc_mode",
    "dlc_std",
    "byte_mean",
    "byte_var",
    "byte_max_change",
    "byte_nunique",
    "byte_entropy",
    "window_fill",
    "time_since_last_seen",
]

# Convenience alias for external callers; v1 is the canonical default.
BEHAVIORAL_FEATURES: list[str] = BEHAVIORAL_FEATURES_V1

# Residual-transformable feature groups (used by ablation experiments).
GROUP_IAT: list[str] = [
    "iat_mean",
    "iat_std",
    "iat_median",
    "iat_min",
    "iat_max",
]
GROUP_FLAT_BYTE: list[str] = [
    "byte_mean",
    "byte_var",
    "byte_max_change",
    "byte_nunique",
    "byte_entropy",
]
GROUP_DLC: list[str] = ["dlc_mode", "dlc_std"]
GROUP_OTHER: list[str] = ["window_fill", "time_since_last_seen"]

# All groups concatenated must equal the v1 set (guard against drift).
assert sorted(GROUP_IAT + GROUP_FLAT_BYTE + GROUP_DLC + GROUP_OTHER) == sorted(
    BEHAVIORAL_FEATURES_V1
), "feature groups must partition v1 set"
