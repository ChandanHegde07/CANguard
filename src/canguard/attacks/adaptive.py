"""White-box adaptive attacks against PIRD (Phase 2).

Threat model
------------
The attacker knows the CAN IDs, the PIRD feature representation, the frozen
per-ID normal statistics ``(mu_k, sigma_k)``, the residual transform
``r = (x - mu_k) / (sigma_k + EPS)``, the detector and its frozen threshold. It
compromises an **existing legitimate ID** and replaces that ID's window feature
vector with

    x_attack = mu_k + alpha * sigma_k * direction        (per targeted feature)

Because residualization is linear in ``x``, this yields residual
``r ~= alpha * direction`` before feasibility clipping. Values are clipped to
physically valid feature ranges and discrete features are rounded, so the
attacker never emits impossible windows.

Leakage
-------
Every statistic used here originates from calibration **normals only**. The
:class:`AttackParameters` object records its provenance and refuses to be built
from anything except ``"calib_normal"``. Attack generation never reads labels,
scores, or statistics derived from the test/attack data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.groups import BEHAVIORAL_FEATURES_V1
from ..features.per_id import GlobalStats, PerIdStats

# ---------------------------------------------------------------------------
# Feasible feature ranges (inclusive) used to clip adversarial values.
# ``None`` means unbounded on that side.
# ---------------------------------------------------------------------------
FEATURE_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "iat_mean": (0.0, None),
    "iat_std": (0.0, None),
    "iat_median": (0.0, None),
    "iat_min": (0.0, None),
    "iat_max": (0.0, None),
    "dlc_mode": (0.0, 8.0),
    "dlc_std": (0.0, 8.0),
    "byte_mean": (0.0, 255.0),
    "byte_var": (0.0, 255.0 * 255.0),
    "byte_max_change": (0.0, 255.0),
    "byte_nunique": (0.0, 240.0),
    "byte_entropy": (0.0, 8.0),
    "window_fill": (0.0, 1.0),
    "time_since_last_seen": (0.0, None),
}

# Features that must remain integers after clipping/rounding.
INTEGER_FEATURES: frozenset[str] = frozenset({"dlc_mode", "byte_nunique"})

# ``window_fill`` is identically 1.0 once a window is full, so it carries no
# residual signal and is not a meaningful attack dimension.
NON_ATTACKABLE: frozenset[str] = frozenset({"window_fill"})

# Worst-case default: every feature that can carry signal.
DEFAULT_ATTACKABLE_FEATURES: list[str] = [
    f for f in BEHAVIORAL_FEATURES_V1 if f not in NON_ATTACKABLE
]

_FEATURE_GROUPS: dict[str, list[str]] = {
    "iat": ["iat_mean", "iat_std", "iat_median", "iat_min", "iat_max"],
    "byte": [
        "byte_mean",
        "byte_var",
        "byte_max_change",
        "byte_nunique",
        "byte_entropy",
    ],
    "dlc": ["dlc_mode", "dlc_std"],
    "other": ["window_fill", "time_since_last_seen"],
}

VALID_DIRECTIONS = ("positive", "negative")
VALID_DRIFT_SHAPES = ("linear", "exponential", "sqrt")
VALID_DRIFT_RATES = ("fast", "medium", "slow")
# mean_shift : x = mu + alpha*sigma*direction   (literal white-box equation)
# additive   : x = x_original + alpha*sigma*direction (modify existing message)
VALID_ATTACK_MODES = ("mean_shift", "additive")

# Fast/medium/slow transition = fraction of the horizon used to ramp to alpha_end.
DRIFT_RATE_FRACTION: dict[str, float] = {
    "fast": 0.25,
    "medium": 0.5,
    "slow": 1.0,
}


# ---------------------------------------------------------------------------
# Attack parameters / provenance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AttackParameters:
    """Frozen, calibration-only per-ID statistics used to generate attacks.

    This is deliberately a thin, explicit object so leakage is auditable: the
    attacker can *only* generate values from these means/stds.
    """

    per_id_stats: Mapping[str, tuple[dict[str, float], dict[str, float]]]
    global_mu: dict[str, float]
    global_sd: dict[str, float]
    feature_cols: tuple[str, ...]
    provenance: str = "calib_normal"
    n_calib_normal_windows: int = 0
    can_ids: tuple[str, ...] = ()

    @classmethod
    def from_stats(
        cls,
        per_id_stats: PerIdStats,
        global_stats: GlobalStats,
        feature_cols: Sequence[str],
        *,
        provenance: str = "calib_normal",
        n_calib_normal_windows: int = 0,
    ) -> AttackParameters:
        if provenance != "calib_normal":
            raise ValueError(
                "AttackParameters may only be built from calibration normals "
                f"(got provenance={provenance!r})."
            )
        gmu, gsd = global_stats
        return cls(
            per_id_stats=per_id_stats,
            global_mu=dict(gmu),
            global_sd=dict(gsd),
            feature_cols=tuple(feature_cols),
            provenance=provenance,
            n_calib_normal_windows=int(n_calib_normal_windows),
            can_ids=tuple(sorted(str(k) for k in per_id_stats)),
        )

    def assert_provenance(self, expected: str = "calib_normal") -> None:
        if self.provenance != expected:
            raise AssertionError(
                f"attack provenance {self.provenance!r} != {expected!r}; "
                "refusing to generate attacks from non-calibration statistics."
            )

    def mean_sd(self, can_id: str) -> tuple[dict[str, float], dict[str, float]]:
        """Return per-ID (mu, sigma) or the global fallback for unseen IDs."""
        cid = str(can_id)
        if cid in self.per_id_stats:
            return self.per_id_stats[cid]
        return self.global_mu, self.global_sd


# ---------------------------------------------------------------------------
# Feature-set resolution
# ---------------------------------------------------------------------------
def resolve_target_features(
    spec: str | Sequence[str] | None,
    feature_cols: Sequence[str] | None = None,
) -> list[str]:
    """Resolve an attack feature spec to a concrete list of feature names.

    ``spec`` may be ``None``/"all_attackable" (worst-case default), a group name
    (``"iat"``/``"byte"``/``"dlc"``/``"other"``), ``"all"``, or an explicit list
    of feature names. The result is intersected with ``feature_cols`` when given.
    """
    if spec is None or spec == "all_attackable":
        feats = list(DEFAULT_ATTACKABLE_FEATURES)
    elif spec == "all":
        feats = list(BEHAVIORAL_FEATURES_V1)
    elif isinstance(spec, str) and spec in _FEATURE_GROUPS:
        feats = list(_FEATURE_GROUPS[spec])
    elif isinstance(spec, (list, tuple)):
        feats = list(spec)
    else:
        raise ValueError(f"Unknown target feature spec: {spec!r}")
    if feature_cols is not None:
        allowed = set(feature_cols)
        feats = [f for f in feats if f in allowed]
    return feats


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def _direction_sign(direction: str) -> float:
    if direction == "positive":
        return 1.0
    if direction == "negative":
        return -1.0
    raise ValueError(f"direction must be one of {VALID_DIRECTIONS}, got {direction!r}")


def attack_feature_matrix(
    params: AttackParameters,
    can_id: str,
    alphas: np.ndarray,
    direction: str,
    target_features: Sequence[str],
    *,
    mode: str = "mean_shift",
    current: np.ndarray | None = None,
) -> np.ndarray:
    """Build the (n, len(target_features)) adversarial feature matrix.

    ``mode="mean_shift"`` follows the literal white-box equation
    ``x = mu + alpha * sigma * sign(direction)``. ``mode="additive"`` instead
    perturbs the *current* legitimate value:
    ``x = x_original + alpha * sigma * sign(direction)``. In both cases values
    are clipped to :data:`FEATURE_BOUNDS` and discrete features are rounded. For
    mean-shift, rows with ``alpha == 0`` keep their original value (severity 0
    is a no-op).
    """
    params.assert_provenance()
    if mode not in VALID_ATTACK_MODES:
        raise ValueError(f"mode must be one of {VALID_ATTACK_MODES}, got {mode!r}")
    mu, sd = params.mean_sd(can_id)
    sign = _direction_sign(direction)
    alphas = np.asarray(alphas, dtype=float)
    feats = list(target_features)
    n = len(alphas)
    if current is not None and current.shape != (n, len(feats)):
        raise ValueError(
            f"current shape {current.shape} != ({n}, {len(feats)})"
        )
    if mode == "additive" and current is None:
        raise ValueError("additive mode requires the current feature matrix")

    q = np.empty((n, len(feats)), dtype=float)
    for j, f in enumerate(feats):
        if mode == "additive":
            base = current[:, j]
            col = base + alphas * sd[f] * sign
        else:  # mean_shift
            base = current[:, j] if current is not None else np.full(n, mu[f])
            col = np.where(alphas == 0.0, base, mu[f] + alphas * sd[f] * sign)
        lo, hi = FEATURE_BOUNDS.get(f, (None, None))
        if lo is not None:
            col = np.maximum(col, lo)
        if hi is not None:
            col = np.minimum(col, hi)
        if f in INTEGER_FEATURES:
            col = np.round(col)
        q[:, j] = col
    return q


def apply_alpha_series(
    params: AttackParameters,
    df: pd.DataFrame,
    target_id: str,
    alphas: Sequence[float],
    direction: str,
    target_features: Sequence[str],
    *,
    mode: str = "mean_shift",
) -> pd.DataFrame:
    """Return a copy of ``df`` with target-ID rows overwritten by attack values.

    Rows are matched by ``can_id == target_id`` in row order; ``alphas`` must
    align with those rows (chronological). Non-target rows are untouched.
    """
    out = df.copy()
    mask = out["can_id"].astype(str).values == str(target_id)
    idx = np.flatnonzero(mask)
    feats = [f for f in target_features if f in out.columns]
    if len(idx) == 0 or len(feats) == 0:
        return out
    alphas = np.asarray(alphas, dtype=float)
    if len(alphas) != len(idx):
        raise ValueError(
            f"alphas length {len(alphas)} != target rows {len(idx)} for id {target_id}"
        )
    arr = out[feats].to_numpy(dtype=float, copy=True)
    current = arr[idx, :]
    q = attack_feature_matrix(
        params,
        target_id,
        alphas,
        direction,
        feats,
        mode=mode,
        current=current,
    )
    arr[idx, :] = q
    out[feats] = arr
    return out


def apply_fixed_attack(
    params: AttackParameters,
    df: pd.DataFrame,
    target_id: str,
    alpha: float,
    direction: str,
    target_features: Sequence[str],
    *,
    mode: str = "mean_shift",
) -> pd.DataFrame:
    """Overwrite every target-ID row with the same severity ``alpha``."""
    n = int((df["can_id"].astype(str).values == str(target_id)).sum())
    return apply_alpha_series(
        params,
        df,
        target_id,
        np.full(n, float(alpha)),
        direction,
        target_features,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Gradual / slow-drift schedules
# ---------------------------------------------------------------------------
def gradual_alphas(
    n: int,
    *,
    alpha_start: float = 0.0,
    alpha_end: float = 1.0,
    shape: str = "linear",
    rate: str = "slow",
    fractions: Sequence[float] | None = None,
) -> np.ndarray:
    """Monotonic schedule of severities for a gradual-drift attack.

    Parameters
    ----------
    n
        Number of windows in the drift horizon.
    alpha_start, alpha_end
        Severity at the start / end of the horizon.
    shape
        ``"linear"``, ``"exponential"`` or ``"sqrt"``.
    rate
        ``"fast"``/``"medium"``/``"slow"``; fraction of the horizon used to
        ramp from 0 to 1 (0.25 / 0.5 / 1.0). Remaining windows stay at
        ``alpha_end``.
    fractions
        Optional explicit percentage schedule (e.g. ``[0.01, 0.02, 0.03, 0.05,
        0.1, 0.2, 1.0]``) linearly spaced across the horizon and scaled by
        ``alpha_end``. Overrides ``shape``/``rate``.

    Returns
    -------
    np.ndarray
        Length-``n`` non-decreasing severity sequence.
    """
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=float)
    if fractions is not None:
        fr = np.maximum.accumulate(np.clip(np.asarray(fractions, dtype=float), 0.0, None))
        if len(fr) == 1:
            frac = np.full(n, float(fr[0]))
        else:
            xi = np.linspace(0.0, 1.0, n)
            frac = np.interp(xi, np.linspace(0.0, 1.0, len(fr)), fr)
    else:
        if shape not in VALID_DRIFT_SHAPES:
            raise ValueError(f"shape must be one of {VALID_DRIFT_SHAPES}, got {shape!r}")
        if rate not in VALID_DRIFT_RATES:
            raise ValueError(f"rate must be one of {VALID_DRIFT_RATES}, got {rate!r}")
        drift_len = max(1, int(round(DRIFT_RATE_FRACTION[rate] * n)))
        denom = max(1, drift_len - 1)
        u = np.clip(np.arange(n, dtype=float) / denom, 0.0, 1.0)
        if shape == "linear":
            frac = u
        elif shape == "sqrt":
            frac = np.sqrt(u)
        else:  # exponential, normalized so frac(1) = 1
            frac = (np.exp(u) - 1.0) / (np.exp(1.0) - 1.0)
    alphas = float(alpha_start) + (float(alpha_end) - float(alpha_start)) * frac
    return np.asarray(alphas, dtype=float)


def gradual_attack(
    params: AttackParameters,
    df: pd.DataFrame,
    target_id: str,
    alphas: Sequence[float],
    direction: str,
    target_features: Sequence[str],
    *,
    mode: str = "mean_shift",
) -> pd.DataFrame:
    """Apply a per-window severity schedule to the target ID's rows."""
    return apply_alpha_series(
        params, df, target_id, alphas, direction, target_features, mode=mode
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
