
from __future__ import annotations

from typing import Any

import pandas as pd

# Canonical versions
PROTOCOL_HCRL_TEMPORAL = "hcrl_temporal_v1"
PROTOCOL_ROAD_PRE_INJECTION = "pre_injection_v1"
PROTOCOL_ROAD_LEGACY_FORBIDDEN = "legacy_all_normals_402040"

PHASE_A = "A"
PHASE_B = "B"
PHASE_C = "C"
PHASE_WORKSHOP = "workshop"

CORRECT_ROAD_PROTOCOLS = frozenset(
    {
        PROTOCOL_ROAD_PRE_INJECTION,
        "pre_injection_v2_inj_priority",  # frame-selection cache key variant; same calib rule
    }
)


def tag_dataframe(
    df: pd.DataFrame,
    *,
    protocol_version: str,
    phase: str,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a copy with protocol_version / phase columns (and optional extras)."""
    out = df.copy()
    out["protocol_version"] = protocol_version
    out["phase"] = phase
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def ensure_protocol_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize protocol → protocol_version if needed."""
    out = df.copy()
    if "protocol_version" not in out.columns and "protocol" in out.columns:
        out["protocol_version"] = out["protocol"].astype(str)
    return out


def assert_road_protocol_ok(df: pd.DataFrame) -> None:

    if df is None or len(df) == 0:
        raise ValueError("Empty ROAD table — cannot verify protocol")

    df = ensure_protocol_column(df)
    if "protocol_version" not in df.columns:
        raise RuntimeError(
            "ROAD table lacks protocol_version (and protocol) metadata. "
            "Refusing to generate manuscript tables from unversioned data "
            "(may be pre-fix Phase-0 numbers)."
        )

    versions = df["protocol_version"].astype(str)
    ok = versions.isin(CORRECT_ROAD_PROTOCOLS)
    if not ok.all():
        vals = sorted(versions[~ok].unique())
        raise RuntimeError(
            f"ROAD table contains non-corrected protocol_version values: {vals}. "
            f"Only {sorted(CORRECT_ROAD_PROTOCOLS)} are allowed for manuscript tables."
        )
