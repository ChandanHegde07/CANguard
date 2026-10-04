"""Automated leakage checks for Phase 4 zero-shot experiments.

These fail loudly rather than allowing a target domain to contaminate fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LeakageError(RuntimeError):
    """Raised when a zero-shot rule would be violated."""


def assert_normal_only(df: pd.DataFrame, *, context: str = "pretraining corpus") -> None:
    if "is_attack" not in df.columns:
        raise LeakageError(f"{context}: missing is_attack column")
    n_attack = int((df["is_attack"] == 1).sum())
    if n_attack > 0:
        raise LeakageError(f"{context} contains {n_attack} attack frames; refusing to fit")


def assert_normal_windows(windows: dict, *, context: str = "pretraining windows") -> None:
    """Every frame of every selected window must be normal."""
    y = windows.get("y")
    if y is None:
        raise LeakageError(f"{context}: missing window labels")
    n_attack = int(np.asarray(y).sum())
    if n_attack > 0:
        raise LeakageError(
            f"{context} contains {n_attack} attack frames inside pretraining windows"
        )


def assert_disjoint(keys_a, keys_b, *, label: str = "captures") -> None:
    overlap = set(map(str, keys_a)) & set(map(str, keys_b))
    if overlap:
        raise LeakageError(f"{label} overlap between source and target: {sorted(overlap)}")


def assert_source_target_disjoint(
    source_domains, target_domains, *, label: str = "domains"
) -> None:
    assert_disjoint(source_domains, target_domains, label=label)


def assert_no_target_fit(target_domains, fitted_domains) -> None:
    """The set of domains that touched fitting must not include the target."""
    bad = set(map(str, target_domains)) & set(map(str, fitted_domains))
    if bad:
        raise LeakageError(f"target domain(s) used in fitting: {sorted(bad)}")


def check_window_integrity(y: np.ndarray) -> None:
    if not np.isfinite(y).all():
        raise LeakageError("window labels contain non-finite values")


__all__ = [
    "LeakageError",
    "assert_disjoint",
    "assert_no_target_fit",
    "assert_normal_only",
    "assert_normal_windows",
    "assert_source_target_disjoint",
    "check_window_integrity",
]
