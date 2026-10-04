"""Model checkpointing for CAN-TFM (weights + tokenizer spec + metadata)."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import CANTFMModel
from .tokenizer import TokenizerConfig


def save_checkpoint(
    path: str | Path, model: CANTFMModel, spec: TokenizerConfig, meta: dict | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    architecture = {
        "d_model": model.d_model,
        "backbone": model.backbone,
        "context": model.context,
        "nhead": model.nhead,
        "num_layers": model.num_layers,
        "dim_feedforward": model.dim_feedforward,
        "dropout": model.dropout,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "spec": spec.to_dict(),
            "meta": meta or {},
            "architecture": architecture,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path, map_location="cpu"
) -> tuple[CANTFMModel, TokenizerConfig, dict]:
    obj = torch.load(Path(path), map_location=map_location, weights_only=False)
    spec = TokenizerConfig(**obj["spec"])
    arch = obj.get("architecture", {})
    model = CANTFMModel(
        spec,
        d_model=int(arch.get("d_model", 128)),
        nhead=int(arch.get("nhead", 4)),
        num_layers=int(arch.get("num_layers", 4)),
        dim_feedforward=int(arch.get("dim_feedforward", 256)),
        dropout=float(arch.get("dropout", 0.1)),
        backbone=str(arch.get("backbone", "transformer")),
        context=int(arch.get("context", 8)),
    )
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, spec, obj.get("meta", {})


__all__ = ["load_checkpoint", "save_checkpoint"]
