"""High-level Phase 4 pipeline: source fitting, frozen scoring, zero-shot eval.

This module is deliberately explicit about what touches fitting. `train_source`
receives only source frames; `score_frames` is evaluation-only and never mutates
model weights or thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from canguard.evaluation.zero_shot import (
    aggregate_frame_scores,
    evaluate_zero_shot_scores,
    sustained_detection_delay,
)
from canguard.exp.cache import FeatureCache

from .data import build_windows_per_capture, load_domain_frames
from .leakage import assert_normal_windows
from .model import build_model
from .scoring import fit_residual_weights, score_next_event
from .tokenizer import TokenizerConfig
from .train import TrainConfig, train_model


@dataclass
class SourceModel:
    model: object
    spec: TokenizerConfig
    weights: np.ndarray
    threshold: float
    source_val_normal_scores: np.ndarray
    train_history: dict
    n_train_windows: int
    n_val_windows: int
    source_val_fpr: float
    source_ids: set = field(default_factory=set)

    def score_frame_scores(self, frames: pd.DataFrame, length: int) -> dict:
        return score_frames(
            self.model,
            frames,
            self.spec,
            length,
            self.weights,
            self.threshold,
            device=_device_of(self.model),
        )


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def split_captures_source_val(
    captures: list[str], val_frac: float = 0.2
) -> tuple[list[str], list[str]]:
    """Hold out whole captures for source validation when possible."""
    caps = sorted(set(map(str, captures)))
    if len(caps) >= 2:
        n_val = max(1, int(round(val_frac * len(caps))))
        n_val = min(n_val, len(caps) - 1)
        return caps[n_val:], caps[:n_val]
    return caps, list(caps)


def load_domain_windows(
    domain: str,
    spec: TokenizerConfig,
    length: int,
    stride: int,
    *,
    roots: dict | None = None,
    max_frames_per_capture: int | None = 200_000,
    captures: list[str] | None = None,
    cache_dir: str | None = ".cache/phase4",
) -> tuple[pd.DataFrame, dict]:
    cache = FeatureCache(cache_dir) if cache_dir else None
    frames = load_domain_frames(
        domain,
        roots=roots,
        max_frames_per_capture=max_frames_per_capture,
        captures=captures,
        cache=cache,
    )
    windows = build_windows_per_capture(frames, spec, length, stride)
    return frames, windows


def _select_normal_windows(windows: dict, frames: pd.DataFrame, captures: list[str]) -> dict:
    """Windows belonging to ``captures`` whose frames are all normal."""
    caps = np.asarray(windows["capture"], dtype=object)
    keep = np.isin(caps, list(map(str, captures)))
    normal = windows["y"].sum(axis=1) == 0
    mask = keep & normal
    m = len(windows["y"])
    return {
        k: (v[mask] if isinstance(v, np.ndarray) and v.shape[0] == m else v)
        for k, v in windows.items()
    }


def train_source(
    source_domains: list[str],
    spec: TokenizerConfig,
    model_cfg: dict,
    train_cfg: TrainConfig,
    length: int,
    *,
    stride: int | None = None,
    roots: dict | None = None,
    max_frames_per_capture: int | None = 200_000,
    captures_by_domain: dict[str, list[str]] | None = None,
    fpr_target: float = 0.01,
    max_train_windows: int | None = None,
    cache_dir: str | None = ".cache/phase4",
) -> SourceModel:
    """Fit CAN-TFM on source normals only (pretraining) and select the threshold."""
    stride = stride or length
    all_train_X, all_train_V = [], []
    val_X, val_V = [], []
    n_train, n_val = 0, 0
    source_ids: set[str] = set()
    for domain in source_domains:
        frames, windows = load_domain_windows(
            domain,
            spec,
            length,
            stride,
            roots=roots,
            max_frames_per_capture=max_frames_per_capture,
            captures=(captures_by_domain or {}).get(domain),
            cache_dir=cache_dir,
        )
        if len(windows["X"]) == 0:
            continue
        source_ids.update(frames["can_id"].astype(str).unique().tolist())
        caps = sorted(set(windows["capture"]))
        train_caps, val_caps = split_captures_source_val(caps)
        train_w = _select_normal_windows(windows, frames, train_caps)
        val_w = _select_normal_windows(windows, frames, val_caps)
        assert_normal_windows(train_w, context=f"{domain} train")
        assert_normal_windows(val_w, context=f"{domain} val")
        all_train_X.append(train_w["X"])
        all_train_V.append(train_w["valid"])
        val_X.append(val_w["X"])
        val_V.append(val_w["valid"])
        n_train += len(train_w["X"])
        n_val += len(val_w["X"])

    if not all_train_X:
        raise RuntimeError("no source normal windows available for pretraining")
    X_train = np.concatenate(all_train_X, axis=0)
    V_train = np.concatenate(all_train_V, axis=0)
    if max_train_windows is not None and len(X_train) > max_train_windows:
        rng = np.random.default_rng(train_cfg.seed)
        sel = rng.choice(len(X_train), size=max_train_windows, replace=False)
        X_train, V_train = X_train[sel], V_train[sel]
    X_val = (
        np.concatenate(val_X, axis=0)
        if val_X
        else np.zeros((0, length, spec.feature_dim), np.float32)
    )
    V_val = np.concatenate(val_V, axis=0) if val_V else np.zeros((0, length, 8), bool)

    model = build_model(model_cfg, spec)
    history = train_model(
        model,
        X_train,
        V_train,
        spec,
        train_cfg,
        X_val=X_val if len(X_val) else None,
        valid_val=V_val if len(X_val) else None,
    )
    weights = fit_residual_weights(
        model,
        X_val if len(X_val) else X_train,
        V_val if len(X_val) else V_train,
        spec,
        device=_device_of(model),
    )
    val_scores = (
        score_next_event(model, X_val, V_val, spec, weights, device=_device_of(model))
        if len(X_val)
        else np.zeros((0, 0))
    )
    flat_val = val_scores.reshape(-1) if val_scores.size else np.zeros(0)
    flat_val = flat_val[np.isfinite(flat_val)]
    threshold = (
        float(np.percentile(flat_val, (1 - fpr_target) * 100)) if flat_val.size else float("inf")
    )
    return SourceModel(
        model=model,
        spec=spec,
        weights=weights,
        threshold=threshold,
        source_val_normal_scores=flat_val,
        train_history=history,
        n_train_windows=n_train,
        n_val_windows=n_val,
        source_val_fpr=float((flat_val >= threshold).mean()) if flat_val.size else float("nan"),
        source_ids=source_ids,
    )


@torch.no_grad()
def score_frames(
    model,
    frames: pd.DataFrame,
    spec: TokenizerConfig,
    length: int,
    weights: np.ndarray,
    threshold: float,
    device=None,
) -> dict:
    """Score every frame of ``frames``. Evaluation-only (no fitting)."""
    device = device or _device_of(model)
    frames = frames.reset_index(drop=True)
    windows = build_windows_per_capture(frames, spec, length, stride=length)
    if len(windows["X"]) == 0:
        n = len(frames)
        return {
            "scores": np.full(n, np.nan),
            "labels": frames["is_attack"].to_numpy(int),
            "times": frames["timestamp"].to_numpy(float),
            "attack_type": (
                frames["attack_type"].to_numpy() if "attack_type" in frames else np.array(["?"] * n)
            ),
            "capture": frames["capture"].to_numpy() if "capture" in frames else np.array(["?"] * n),
            "threshold": threshold,
        }
    wscores = score_next_event(model, windows["X"], windows["valid"], spec, weights, device=device)
    frame_scores, _counts = aggregate_frame_scores(wscores, windows["frame_row"], len(frames))
    return {
        "scores": frame_scores,
        "labels": frames["is_attack"].to_numpy(int),
        "times": frames["timestamp"].to_numpy(float),
        "attack_type": (
            frames["attack_type"].to_numpy()
            if "attack_type" in frames
            else np.array(["?"] * len(frames))
        ),
        "capture": (
            frames["capture"].to_numpy() if "capture" in frames else np.array(["?"] * len(frames))
        ),
        "threshold": threshold,
        "window_capture": windows["capture"],
    }


def evaluate_source_on_target(
    source: SourceModel,
    target_frames: pd.DataFrame,
    length: int,
    *,
    fpr_target: float = 0.01,
    nominal_fprs: tuple[float, ...] = (0.01, 0.05),
) -> dict:
    scored = score_frames(
        source.model, target_frames, source.spec, length, source.weights, source.threshold
    )
    result = evaluate_zero_shot_scores(
        source.source_val_normal_scores,
        scored["scores"],
        scored["labels"],
        fpr_target=fpr_target,
        nominal_fprs=nominal_fprs,
    )
    finite = np.isfinite(scored["scores"])
    delay = sustained_detection_delay(
        scored["times"][finite],
        scored["scores"][finite],
        scored["labels"][finite],
        source.threshold,
    )
    result.update({"delay_" + k: v for k, v in delay.items()})
    return result


def unseen_id_subset(
    frames: pd.DataFrame, source_ids: set[str], mode: str = "unseen"
) -> pd.DataFrame:
    """Restrict target frames to IDs unseen (or seen) in the source domain."""
    canonical = frames["can_id"].astype(str)
    if mode == "unseen":
        return frames[~canonical.isin(source_ids)].reset_index(drop=True)
    return frames[canonical.isin(source_ids)].reset_index(drop=True)


# Backwards-compatible aliases used by runners.
def train_source_model(**kwargs) -> SourceModel:
    return train_source(**kwargs)


__all__ = [
    "SourceModel",
    "evaluate_source_on_target",
    "score_frames",
    "split_captures_source_val",
    "train_source",
    "train_source_model",
    "unseen_id_subset",
]
