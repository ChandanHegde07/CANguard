"""Anomaly scoring for CAN-TFM.

Primary score: **causal next-event residual**. At each frame the model predicts
the next event from its past; the weighted reconstruction error of the *actual*
next event is the anomaly score. Weights are per-dimension source-normal residual
standard deviations (source-only), so no target statistics enter scoring.

Diagnostic score: diagonal-Mahalanobis distance of the contextual representation
to the source-normal embedding distribution (source-only).
"""

from __future__ import annotations

import numpy as np
import torch

from .tokenizer import TokenizerConfig


def _residual_matrix(
    preds: dict[str, torch.Tensor],
    X: torch.Tensor,
    valid: torch.Tensor,
    spec: TokenizerConfig,
) -> torch.Tensor:
    """Next-event residual vector [B,L-1,D] with invalid payload dims zeroed."""
    ps, pe = 0, spec.payload_dim
    ts, te = pe + 1, pe + 2
    pred_payload = preds["payload"][:, :-1]
    x_payload = X[:, 1:, ps:pe]
    vnext = valid[:, 1:]
    r_payload = (pred_payload - x_payload) * vnext
    pred_timing = preds["timing"][:, :-1]
    x_timing = X[:, 1:, ts:te]
    r_timing = pred_timing - x_timing
    return torch.cat([r_payload, r_timing], dim=-1)


def _valid_matrix(valid: torch.Tensor, spec: TokenizerConfig) -> torch.Tensor:
    vnext = valid[:, 1:]
    timing = torch.ones((*vnext.shape[:-1], 1), dtype=vnext.dtype, device=vnext.device)
    return torch.cat([vnext, timing], dim=-1)


@torch.no_grad()
def fit_residual_weights(
    model,
    X: np.ndarray,
    valid: np.ndarray,
    spec: TokenizerConfig,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    eps: float = 1e-3,
) -> np.ndarray:
    """Per-dimension inverse residual std from source-normal windows (source-only)."""
    model.eval()
    resids = []
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Vt = torch.as_tensor(valid, dtype=torch.bool)
    for i in range(0, len(Xt), batch_size):
        xb, vb = Xt[i : i + batch_size].to(device), Vt[i : i + batch_size].to(device)
        _, preds = model(xb, causal=True)
        r = _residual_matrix(preds, xb, vb, spec)
        m = _valid_matrix(vb, spec)
        resids.append((r * m).reshape(-1, r.shape[-1]).cpu().numpy())
    if not resids:
        return np.ones(spec.payload_dim + 1, dtype=np.float32)
    R = np.concatenate(resids, axis=0)
    std = R.std(axis=0)
    std = np.where(std < eps, eps, std)
    return (1.0 / std).astype(np.float32)


@torch.no_grad()
def score_next_event(
    model,
    X: np.ndarray,
    valid: np.ndarray,
    spec: TokenizerConfig,
    weights: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Return per-position next-event scores ``[M, L-1]`` (position 0 excluded)."""
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Vt = torch.as_tensor(valid, dtype=torch.bool)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(Xt), batch_size):
        xb, vb = Xt[i : i + batch_size].to(device), Vt[i : i + batch_size].to(device)
        _, preds = model(xb, causal=True)
        r = _residual_matrix(preds, xb, vb, spec)
        m = _valid_matrix(vb, spec)
        score = (((r * w) ** 2) * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1)
        out.append(score.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


@torch.no_grad()
def embed_positions(
    model,
    X: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    causal: bool = False,
) -> np.ndarray:
    """Contextual representations ``[M, L, d]`` for every position.

    ``causal=True`` restricts each position to its past (online/streaming
    protocol); ``causal=False`` is bidirectional (non-causal pretraining).
    The information available at inference is reported explicitly.
    """
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    chunks = []
    for i in range(0, len(Xt), batch_size):
        z, _ = model(Xt[i : i + batch_size].to(device), causal=causal)
        chunks.append(z.cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0, 0), np.float32)


@torch.no_grad()
def score_representation(
    model,
    X: np.ndarray,
    mu: np.ndarray,
    inv_var: np.ndarray,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Diagonal-Mahalanobis score of contextual embeddings ``[M, L]``."""
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device)
    iv_t = torch.as_tensor(inv_var, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(Xt), batch_size):
        z, _ = model(Xt[i : i + batch_size].to(device), causal=False)
        d = ((z - mu_t) ** 2 * iv_t).mean(dim=-1)
        out.append(d.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), np.float32)


def fit_representation_stats(model, X: np.ndarray, device="cpu", batch_size: int = 256):
    """Source-normal embedding mean and inverse variance (source-only)."""
    Z = embed_positions(model, X, device=device, batch_size=batch_size)  # [M,L,d]
    flat = Z.reshape(-1, Z.shape[-1])
    mu = flat.mean(axis=0)
    var = flat.var(axis=0)
    var = np.where(var < 1e-6, 1e-6, var)
    return mu.astype(np.float32), (1.0 / var).astype(np.float32)


__all__ = [
    "fit_representation_stats",
    "fit_residual_weights",
    "score_next_event",
    "score_representation",
]
