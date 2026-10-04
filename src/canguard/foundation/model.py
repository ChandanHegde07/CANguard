"""CAN-TFM backbones: causal next-event prediction and masked event modeling.

All backbones expose the same interface so experiments are comparable:

    forward(x, mask=None, causal=True) -> (z, preds)

where ``z`` is the contextual representation ``[B,L,d]`` and ``preds`` is a dict
of per-position predictions for the features at that position. Next-event
objectives compare ``preds[:, :-1]`` to ``x[:, 1:]``; masked objectives compare
``preds[mask]`` to the true features at masked positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .tokenizer import TokenizerConfig

BACKBONES = ("transformer", "gru", "mlp")


def _sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)  # [1,max_len,d]


class CANTFMModel(nn.Module):
    def __init__(
        self,
        spec: TokenizerConfig,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        backbone: str = "transformer",
        context: int = 8,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}")
        self.spec = spec
        self.backbone = backbone
        self.d_model = d_model
        self.context = context
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.in_proj = nn.Linear(spec.feature_dim, d_model)
        self.register_buffer("pos", _sinusoidal(max_len, d_model), persistent=False)

        if backbone == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder: nn.Module = nn.TransformerEncoder(layer, num_layers=num_layers)
        elif backbone == "gru":
            self.encoder = nn.GRU(
                d_model,
                d_model,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:  # mlp (causal context MLP)
            self.encoder = nn.Sequential(
                nn.Linear(spec.feature_dim * context, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )

        self.payload_head = nn.Linear(d_model, spec.payload_dim)
        self.timing_head = nn.Linear(d_model, 1)
        self.id_head = nn.Linear(d_model, spec.id_dim) if spec.id_dim else None

    def _mlp_context(self, x: torch.Tensor) -> torch.Tensor:
        b, length, f = x.shape
        k = min(self.context, length)
        padded = torch.nn.functional.pad(x, (0, 0, k - 1, 0))  # left pad
        ctx = padded.unfold(1, k, 1)  # [B,L,k,F]
        ctx = ctx.reshape(b, length, k * f)
        return self.encoder(ctx)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, causal: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, length, _ = x.shape
        inp = x if mask is None else x.masked_fill(mask.unsqueeze(-1), 0.0)
        h = self.in_proj(inp)

        if self.backbone == "transformer":
            h = h + self.pos[:, :length]
            src_mask = None
            if causal:
                src_mask = torch.triu(
                    torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
                )
            z = self.encoder(h, mask=src_mask)
        elif self.backbone == "gru":
            z, _ = self.encoder(h)
        else:
            z = self._mlp_context(inp)

        preds = {"payload": self.payload_head(z), "timing": self.timing_head(z)}
        if self.id_head is not None:
            preds["id"] = self.id_head(z)
        return z, preds

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict, spec: TokenizerConfig) -> CANTFMModel:
    """Build a model from a config dict (see experiments/phase4/configs)."""
    m = cfg.get("model", cfg)
    return CANTFMModel(
        spec,
        d_model=int(m.get("d_model", 128)),
        nhead=int(m.get("nhead", 4)),
        num_layers=int(m.get("num_layers", 4)),
        dim_feedforward=int(m.get("dim_feedforward", 256)),
        dropout=float(m.get("dropout", 0.1)),
        backbone=str(m.get("backbone", "transformer")),
        context=int(m.get("context", 8)),
    )


__all__ = ["BACKBONES", "CANTFMModel", "build_model"]
