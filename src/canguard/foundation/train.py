"""Self-supervised training loop for CAN-TFM (normal traffic only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .objectives import masked_loss, next_event_loss, sample_mask
from .tokenizer import TokenizerConfig


@dataclass
class TrainConfig:
    objective: str = "next_event"  # next_event | masked | combined
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    mask_prob: float = 0.15
    lambda_timing: float = 1.0
    lambda_id: float = 1.0
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


def _pick_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def train_model(
    model,
    X: np.ndarray,
    valid: np.ndarray,
    spec: TokenizerConfig,
    cfg: TrainConfig,
    X_val: np.ndarray | None = None,
    valid_val: np.ndarray | None = None,
) -> dict:
    """Train ``model`` in place; return a history dict."""
    device = _pick_device(cfg.device)
    model.to(device)
    model.train()

    gen = torch.Generator().manual_seed(int(cfg.seed))
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Vt = torch.as_tensor(valid, dtype=torch.bool)
    n = len(Xt)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        epoch_losses = []
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb, vb = Xt[idx].to(device), Vt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            if cfg.objective == "next_event":
                _, preds = model(xb, causal=True)
                loss = next_event_loss(
                    preds,
                    xb,
                    vb,
                    _slices(spec),
                    lambda_timing=cfg.lambda_timing,
                    lambda_id=cfg.lambda_id,
                )
            elif cfg.objective == "masked":
                mask = sample_mask(xb.shape[:2], cfg.mask_prob, device, gen)
                _, preds = model(xb, mask=mask, causal=False)
                loss = masked_loss(
                    preds,
                    xb,
                    vb,
                    _slices(spec),
                    mask,
                    lambda_timing=cfg.lambda_timing,
                    lambda_id=cfg.lambda_id,
                )
            else:  # combined
                _, preds = model(xb, causal=True)
                l1 = next_event_loss(
                    preds,
                    xb,
                    vb,
                    _slices(spec),
                    lambda_timing=cfg.lambda_timing,
                    lambda_id=cfg.lambda_id,
                )
                mask = sample_mask(xb.shape[:2], cfg.mask_prob, device, gen)
                _, preds2 = model(xb, mask=mask, causal=False)
                l2 = masked_loss(
                    preds2,
                    xb,
                    vb,
                    _slices(spec),
                    mask,
                    lambda_timing=cfg.lambda_timing,
                    lambda_id=cfg.lambda_id,
                )
                loss = l1 + l2
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history["train_loss"].append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
        if X_val is not None and len(X_val):
            history["val_loss"].append(_eval_loss(model, X_val, valid_val, spec, cfg, device))
    model.eval()
    return history


@torch.no_grad()
def _eval_loss(model, X, valid, spec, cfg: TrainConfig, device) -> float:
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    Vt = torch.as_tensor(valid, dtype=torch.bool)
    losses = []
    for i in range(0, len(Xt), cfg.batch_size):
        xb, vb = Xt[i : i + cfg.batch_size].to(device), Vt[i : i + cfg.batch_size].to(device)
        _, preds = model(xb, causal=True)
        losses.append(float(next_event_loss(preds, xb, vb, _slices(spec)).cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _slices(spec: TokenizerConfig) -> dict[str, tuple[int, int]]:
    p = spec.payload_dim
    return {
        "payload": (0, p),
        "dlc": (p, p + 1),
        "timing": (p + 1, p + 2),
        "id": (p + 2, p + 2 + spec.id_dim),
    }


__all__ = ["TrainConfig", "train_model"]
