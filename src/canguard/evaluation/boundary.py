"""Detection-boundary analysis for Phase 2 adaptive attacks.

Given per-configuration detection outcomes (flagged / not flagged) as a function
of attack severity ``alpha``, this module computes detection probabilities with
Wilson confidence intervals and the severities at which detection probability
crosses operational thresholds (e.g. 50%, 90%).

No result is forced: if the pre-registered severity grid does not bracket a
crossing, the crossing is reported as ``None`` with an explicit reason
(``floor_exceeded`` / ``ceiling_not_reached``).
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_LEVELS: tuple[float, ...] = (0.5, 0.9)

_REASON_FLOOR = "floor_exceeded"
_REASON_CEILING = "ceiling_not_reached"
_REASON_NO_BRACKET = "not_bracketed"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def detection_probability(flags: np.ndarray, z: float = 1.96) -> dict[str, float]:
    """Detection probability (= recall on injected windows) with a Wilson CI."""
    flags = np.asarray(flags).astype(int)
    n = int(flags.size)
    k = int(flags.sum())
    lo, hi = wilson_interval(k, n, z=z)
    return {
        "n": n,
        "n_detected": k,
        "detection_probability": (k / n) if n else float("nan"),
        "detection_prob_ci_low": lo,
        "detection_prob_ci_high": hi,
    }


def crossing_severity(
    severities: np.ndarray,
    probabilities: np.ndarray,
    level: float,
) -> tuple[float | None, str]:
    """Severity at which detection probability first reaches ``level``.

    Uses linear interpolation between the first bracketing grid points. Returns
    ``(None, reason)`` when the grid does not bracket the crossing.
    """
    severities = np.asarray(severities, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(severities)
    severities = severities[order]
    probabilities = probabilities[order]
    if severities.size == 0:
        return None, _REASON_NO_BRACKET
    if probabilities[0] >= level:
        return None, _REASON_FLOOR
    if probabilities[-1] < level:
        return None, _REASON_CEILING
    for i in range(severities.size - 1):
        p0, p1 = probabilities[i], probabilities[i + 1]
        if p0 < level <= p1:
            if p1 <= p0:
                return float(severities[i]), ""
            frac = (level - p0) / (p1 - p0)
            return float(severities[i] + frac * (severities[i + 1] - severities[i])), ""
    return None, _REASON_NO_BRACKET


def summarize_boundary(
    severities: np.ndarray,
    probabilities: np.ndarray,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
) -> dict[str, float | None | str]:
    """Compute alpha at each operational detection-probability level."""
    out: dict[str, float | None | str] = {}
    for level in levels:
        tag = f"alpha_{int(round(level * 100))}"
        value, reason = crossing_severity(severities, probabilities, level)
        out[tag] = value
        out[f"{tag}_reason"] = reason if value is None else ""
    return out


__all__ = [
    "DEFAULT_LEVELS",
    "crossing_severity",
    "detection_probability",
    "summarize_boundary",
    "wilson_interval",
]
