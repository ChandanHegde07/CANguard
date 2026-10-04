"""Self-supervised objectives for CAN-TFM (normal traffic only).

Two objectives, individually or combined:

* ``next_event``   — predict token ``t+1`` from tokens ``<= t`` (causal).
* ``masked``       — reconstruct randomly masked tokens from full context
                     (bidirectional).

Losses are (valid-byte-weighted) MSE on payload + timing (+ ID if represented).
ID is deliberately excluded from the *anomaly score* later; it is only a
reconstruction target when an ID representation is used.
"""

from __future__ import annotations

import torch

OBJECTIVES = ("next_event", "masked", "combined")


def sample_mask(shape, prob: float, device: torch.device, generator=None) -> torch.Tensor:
    if prob <= 0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    return torch.rand(shape, device=device, generator=generator) < prob


def _expand_valid(valid: torch.Tensor, payload_mode: str) -> torch.Tensor:
    # valid is [B,L,8]; expand to bit layout if needed.
    return valid.repeat_interleave(8, dim=-1) if payload_mode == "bit" else valid


def _mse(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.zeros((), device=a.device)
    return (((a - b) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def reconstruction_terms(
    preds: dict[str, torch.Tensor],
    x: torch.Tensor,
    valid: torch.Tensor,
    slices: dict[str, tuple[int, int]],
    *,
    position_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted payload/timing/id reconstruction losses.

    ``position_mask`` [B,L] bool selects which positions contribute (None = all).
    """
    ps, pe = slices["payload"]
    ts, te = slices["timing"]
    x_payload = x[..., ps:pe]
    x_timing = x[..., ts:te]
    pv = valid  # [B,L,payload_dim]
    if position_mask is not None:
        x_payload = x_payload[position_mask]
        x_timing = x_timing[position_mask]
        pred_payload = preds["payload"][position_mask]
        pred_timing = preds["timing"][position_mask]
        pv = valid[position_mask]
    else:
        pred_payload = preds["payload"]
        pred_timing = preds["timing"]
    payload_loss = _mse(pv, pred_payload, x_payload)
    timing_loss = _mse(torch.ones_like(x_timing, dtype=torch.bool), pred_timing, x_timing)
    out = {"payload_loss": payload_loss, "timing_loss": timing_loss}
    if "id" in preds:
        is_, ie = slices["id"]
        x_id = x[..., is_:ie]
        pred_id = preds["id"]
        if position_mask is not None:
            x_id = x_id[position_mask]
            pred_id = pred_id[position_mask]
        out["id_loss"] = _mse(torch.ones_like(x_id, dtype=torch.bool), pred_id, x_id)
    return out


def next_event_loss(
    preds: dict[str, torch.Tensor],
    x: torch.Tensor,
    valid: torch.Tensor,
    slices: dict[str, tuple[int, int]],
    *,
    lambda_timing: float = 1.0,
    lambda_id: float = 1.0,
) -> torch.Tensor:
    """Predict token t+1 from context <= t (align predictions left by one)."""
    shifted_preds = {k: v[:, :-1] for k, v in preds.items()}
    x_next = x[:, 1:]
    valid_next = valid[:, 1:]
    terms = reconstruction_terms(shifted_preds, x_next, valid_next, slices)
    loss = terms["payload_loss"] + lambda_timing * terms["timing_loss"]
    if "id_loss" in terms:
        loss = loss + lambda_id * terms["id_loss"]
    return loss


def masked_loss(
    preds: dict[str, torch.Tensor],
    x: torch.Tensor,
    valid: torch.Tensor,
    slices: dict[str, tuple[int, int]],
    mask: torch.Tensor,
    *,
    lambda_timing: float = 1.0,
    lambda_id: float = 1.0,
) -> torch.Tensor:
    terms = reconstruction_terms(preds, x, valid, slices, position_mask=mask)
    loss = terms["payload_loss"] + lambda_timing * terms["timing_loss"]
    if "id_loss" in terms:
        loss = loss + lambda_id * terms["id_loss"]
    return loss


__all__ = [
    "OBJECTIVES",
    "masked_loss",
    "next_event_loss",
    "reconstruction_terms",
    "sample_mask",
]
