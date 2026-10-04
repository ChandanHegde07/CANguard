"""Phase 4 — ID-agnostic CAN temporal foundation model (CAN-TFM).

Self-supervised temporal representation learning on normal CAN traffic for
zero-shot, ID-agnostic intrusion detection. Kept additive; Phase 2/3 untouched.
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .model import build_model
from .scoring import score_next_event, score_representation
from .tokenizer import TokenizerConfig, frames_to_features, make_windows

__all__ = [
    "TokenizerConfig",
    "build_model",
    "frames_to_features",
    "load_checkpoint",
    "make_windows",
    "save_checkpoint",
    "score_next_event",
    "score_representation",
]
