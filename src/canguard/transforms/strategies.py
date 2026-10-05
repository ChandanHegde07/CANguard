"""Pluggable residualization strategies for PIRD validation (Phase 1).

The behavioral detector operates on a per-ID window feature table. Raw window
features live on wildly different scales (inter-arrival times vs DLC vs byte
statistics), so the choice of *representation* is a first-class experimental
variable. This module exposes a small, uniform interface so that experiments can
swap residualization strategies without touching detector or evaluation code.

Strategies share one contract: ``fit(calib_normals, feature_cols)`` learns the
normal reference (from attack-free calibration windows only) and
``transform(df, feature_cols)`` returns a new frame whose ``f + suffix`` columns
hold the transformed values, preserving ``can_id`` / ``timestamp`` /
``is_attack`` for downstream evaluation.

Implemented strategies:

* ``raw``            — identity (no residualization).
* ``global_z``       — global standardization: ``(x - mu) / sd``.
* ``per_id_z``       — per-ID z-score residual (the PIRD operator).
* ``per_id_mad``     — robust per-ID residual using median / MAD.
* ``per_id_quantile``— per-ID empirical-CDF → standard-normal quantile map.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..features.per_id import (
    EPS,
    MIN_WINDOWS_PER_ID,
    fit_per_id_stats,
    transform_residuals,
)

# Consistency constant so that MAD estimates the std of a normal distribution.
MAD_SCALE = 1.4826
# Quantile map is clipped away from the 0/1 boundary to keep the normal PPF finite.
_QUANTILE_EPS = 1e-6


def _normals(df_calib: pd.DataFrame) -> pd.DataFrame:
    return df_calib[df_calib["is_attack"] == 0]


def _assemble(
    df: pd.DataFrame,
    R: np.ndarray,
    feature_cols: list[str],
    suffix: str,
) -> pd.DataFrame:
    """Attach transformed feature columns plus identity/label columns."""
    out: dict[str, np.ndarray] = {f + suffix: R[:, j] for j, f in enumerate(feature_cols)}
    out["can_id"] = df["can_id"].astype(str).values
    out["timestamp"] = df["timestamp"].values
    out["is_attack"] = df["is_attack"].values
    if "attack_frac" in df.columns:
        out["attack_frac"] = df["attack_frac"].values
    else:
        out["attack_frac"] = np.zeros(len(df), dtype=float)
    # Carry labels used by per-attack / capture evaluation (not model inputs).
    for extra in ("attack_type", "capture"):
        if extra in df.columns:
            out[extra] = df[extra].values
    return pd.DataFrame(out)


class BaseResidualizer(ABC):
    """Uniform fit/transform interface for window representations."""

    name: str = "base"
    _feature_cols: list[str]

    @abstractmethod
    def fit(self, calib_df: pd.DataFrame, feature_cols: list[str]) -> BaseResidualizer:
        raise NotImplementedError

    @abstractmethod
    def transform(
        self, df: pd.DataFrame, feature_cols: list[str], suffix: str = "_res"
    ) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(
        self,
        calib_df: pd.DataFrame,
        df: pd.DataFrame,
        feature_cols: list[str],
        suffix: str = "_res",
    ) -> pd.DataFrame:
        self.fit(calib_df, feature_cols)
        return self.transform(df, feature_cols, suffix)

    @property
    def residual_cols(self) -> list[str]:
        return [c + "_res" for c in self._feature_cols]


def _apply_per_id(
    df: pd.DataFrame,
    feature_cols: list[str],
    per_id_loc: dict[str, np.ndarray],
    global_loc: np.ndarray,
    per_id_scale: dict[str, np.ndarray],
    global_scale: np.ndarray,
) -> np.ndarray:
    ids = df["can_id"].astype(str).values
    X = df[feature_cols].to_numpy(dtype=float, copy=True)
    R = np.empty_like(X)
    for cid in pd.unique(ids):
        mask = ids == cid
        loc = per_id_loc.get(cid, global_loc)
        scale = per_id_scale.get(cid, global_scale)
        R[mask] = (X[mask] - loc) / (scale + EPS)
    R = np.where(np.isnan(R), 0.0, R)
    return R


class RawResidualizer(BaseResidualizer):
    """Identity transform: expose raw features under the ``*_res`` schema."""

    name = "raw"

    def fit(self, calib_df: pd.DataFrame, feature_cols: list[str]) -> RawResidualizer:
        self._feature_cols = list(feature_cols)
        return self

    def transform(
        self, df: pd.DataFrame, feature_cols: list[str], suffix: str = "_res"
    ) -> pd.DataFrame:
        self._feature_cols = list(feature_cols)
        R = df[feature_cols].to_numpy(dtype=float, copy=True)
        R = np.where(np.isnan(R), 0.0, R)
        return _assemble(df, R, feature_cols, suffix)


class GlobalZResidualizer(BaseResidualizer):
    """Global standardization fitted on all attack-free calibration windows."""

    name = "global_z"

    def fit(self, calib_df: pd.DataFrame, feature_cols: list[str]) -> GlobalZResidualizer:
        self._feature_cols = list(feature_cols)
        normals = _normals(calib_df)
        block = normals[feature_cols].to_numpy(dtype=float)
        self.mean_ = np.nanmean(block, axis=0)
        self.std_ = np.nanstd(block, axis=0)
        return self

    def transform(
        self, df: pd.DataFrame, feature_cols: list[str], suffix: str = "_res"
    ) -> pd.DataFrame:
        X = df[feature_cols].to_numpy(dtype=float, copy=True)
        with np.errstate(invalid="ignore"):
            R = (X - self.mean_) / (self.std_ + EPS)
        R = np.where(np.isnan(R), 0.0, R)
        return _assemble(df, R, feature_cols, suffix)


class PerIdZResidualizer(BaseResidualizer):
    """Per-ID z-score residual — the canonical PIRD operator.

    Delegates to :func:`canguard.features.per_id.transform_residuals` so that the
    Phase 1 experiments reproduce the existing library behavior exactly.
    """

    name = "per_id_z"

    def fit(self, calib_df: pd.DataFrame, feature_cols: list[str]) -> PerIdZResidualizer:
        self._feature_cols = list(feature_cols)
        self.stats_, self.global_stats_ = fit_per_id_stats(calib_df, feature_cols)
        return self

    def transform(
        self, df: pd.DataFrame, feature_cols: list[str], suffix: str = "_res"
    ) -> pd.DataFrame:
        out = transform_residuals(
            df, self.stats_, self.global_stats_, feature_cols, suffix=suffix
        )
        # transform_residuals drops label columns; re-attach for per-attack eval.
        for extra in ("attack_type", "capture"):
            if extra in df.columns:
                out[extra] = df[extra].values
        return out


class PerIdMADResidualizer(BaseResidualizer):
    """Robust per-ID residual: ``(x - median) / (1.4826 * MAD + eps)``."""

    name = "per_id_mad"

    def fit(self, calib_df: pd.DataFrame, feature_cols: list[str]) -> PerIdMADResidualizer:
        self._feature_cols = list(feature_cols)
        normals = _normals(calib_df)
        ids = normals["can_id"].astype(str).values
        X = normals[feature_cols].to_numpy(dtype=float)

        def _mad(block: np.ndarray) -> np.ndarray:
            med = np.nanmedian(block, axis=0)
            return MAD_SCALE * np.nanmedian(np.abs(block - med), axis=0)

        self.global_med_ = np.nanmedian(X, axis=0)
        self.global_mad_ = _mad(X)
        self.per_id_med_: dict[str, np.ndarray] = {}
        self.per_id_mad_: dict[str, np.ndarray] = {}
        for cid in pd.unique(ids):
            mask = ids == cid
            if mask.sum() < MIN_WINDOWS_PER_ID:
                continue
            block = X[mask]
            self.per_id_med_[cid] = np.nanmedian(block, axis=0)
            self.per_id_mad_[cid] = _mad(block)
        return self

    def transform(
        self, df: pd.DataFrame, feature_cols: list[str], suffix: str = "_res"
    ) -> pd.DataFrame:
        R = _apply_per_id(
            df,
            feature_cols,
            self.per_id_med_,
            self.global_med_,
            self.per_id_mad_,
            self.global_mad_,
        )
        return _assemble(df, R, feature_cols, suffix)


class PerIdQuantileResidualizer(BaseResidualizer):
    """Per-ID empirical-CDF → standard-normal quantile (rank/Gaussianization).

    For each ID and feature, calibration normals define the empirical CDF. A new
    value maps to ``Phi^{-1}(rank / (n + 1))``, so a window whose feature is at
    the median of its ID becomes 0 and one at the tail becomes strongly
    positive/negative. Unseen or data-poor IDs fall back to the global CDF.
    """

    name = "per_id_quantile"

    def fit(self, calib_df: pd.DataFrame, feature_cols: list[str]) -> PerIdQuantileResidualizer:
        self._feature_cols = list(feature_cols)
        normals = _normals(calib_df)
        ids = normals["can_id"].astype(str).values
        X = normals[feature_cols].to_numpy(dtype=float)
        self.global_sorted_ = np.sort(X, axis=0)
        self.per_id_sorted_: dict[str, np.ndarray] = {}
        for cid in pd.unique(ids):
            mask = ids == cid
            if mask.sum() < MIN_WINDOWS_PER_ID:
                continue
            self.per_id_sorted_[cid] = np.sort(X[mask], axis=0)
        return self

    @staticmethod
    def _map(block: np.ndarray, reference: np.ndarray) -> np.ndarray:
        n, d = block.shape
        out = np.zeros((n, d), dtype=float)
        for j in range(d):
            col = reference[:, j]
            finite = np.isfinite(col)
            col = col[finite]
            m = len(col)
            if m == 0:
                out[:, j] = 0.0
                continue
            xj = block[:, j]
            ok = np.isfinite(xj)
            u = np.full(n, 0.5, dtype=float)
            ranks = np.searchsorted(col, xj[ok], side="right")
            u[ok] = ranks / (m + 1.0)
            out[:, j] = norm.ppf(np.clip(u, _QUANTILE_EPS, 1.0 - _QUANTILE_EPS))
        return out

    def transform(
        self, df: pd.DataFrame, feature_cols: list[str], suffix: str = "_res"
    ) -> pd.DataFrame:
        ids = df["can_id"].astype(str).values
        X = df[feature_cols].to_numpy(dtype=float, copy=True)
        R = np.empty_like(X)
        for cid in pd.unique(ids):
            mask = ids == cid
            ref = self.per_id_sorted_.get(cid, self.global_sorted_)
            R[mask] = self._map(X[mask], ref)
        R = np.where(np.isnan(R), 0.0, R)
        return _assemble(df, R, feature_cols, suffix)


RESIDUALIZERS: dict[str, type[BaseResidualizer]] = {
    "raw": RawResidualizer,
    "global_z": GlobalZResidualizer,
    "per_id_z": PerIdZResidualizer,
    "per_id_mad": PerIdMADResidualizer,
    "per_id_quantile": PerIdQuantileResidualizer,
}

DEFAULT_ORDER: list[str] = [
    "raw",
    "global_z",
    "per_id_z",
    "per_id_mad",
    "per_id_quantile",
]


def list_residualizers() -> list[str]:
    return sorted(RESIDUALIZERS)


def build_residualizer(name: str) -> BaseResidualizer:
    key = str(name).lower().strip()
    if key not in RESIDUALIZERS:
        raise ValueError(
            f"Unknown residualizer '{name}'. Available: {list_residualizers()}"
        )
    return RESIDUALIZERS[key]()
