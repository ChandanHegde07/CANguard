"""CAN frame tokenization and ID-agnostic feature representation.

Every transform here is **fixed** (no statistics fitted on data), so a target
domain contributes nothing to representation. This is what makes the later
zero-shot protocol valid.

Feature vector per CAN frame (order fixed):

    [ payload (8 bytes or 64 bits) | dlc | timing | id-repr ]

* payload: value/255 in byte mode, raw bits in bit mode.
* dlc: dlc/8, clipped to [0, 1].
* timing: log1p(dt_ms) / log1p(dt_scale_ms), clipped to [0, 1].
* id-repr: depends on ``id_mode``:
    - ``none``          : omitted entirely (ID-free ablation);
    - ``numeric``       : id / 2**id_bits;
    - ``numeric_embed`` : same scalar (projected by the model);
    - ``bit``           : ``id_bits`` bit vector (leading zeros for 11-bit IDs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

BYTE_COLS = [f"data_{i}" for i in range(8)]
VALID_PAYLOAD_MODES = ("byte", "bit")
VALID_ID_MODES = ("none", "numeric", "numeric_embed", "bit")


@dataclass(frozen=True)
class TokenizerConfig:
    payload_mode: str = "byte"
    id_mode: str = "none"
    id_bits: int = 29
    dt_scale_ms: float = 1000.0

    def __post_init__(self) -> None:
        if self.payload_mode not in VALID_PAYLOAD_MODES:
            raise ValueError(f"payload_mode must be one of {VALID_PAYLOAD_MODES}")
        if self.id_mode not in VALID_ID_MODES:
            raise ValueError(f"id_mode must be one of {VALID_ID_MODES}")
        if self.id_bits not in (11, 29):
            raise ValueError("id_bits must be 11 or 29")

    @property
    def payload_dim(self) -> int:
        return 8 if self.payload_mode == "byte" else 8 * 8

    @property
    def id_dim(self) -> int:
        if self.id_mode == "none":
            return 0
        if self.id_mode == "bit":
            return self.id_bits
        return 1  # numeric / numeric_embed

    @property
    def feature_dim(self) -> int:
        return self.payload_dim + 1 + 1 + self.id_dim

    def to_dict(self) -> dict:
        return asdict(self)


def _hex_byte(value) -> int:
    if value is None:
        return 0
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return 0
    try:
        return int(s, 16) & 0xFF
    except ValueError:
        return 0


def _canonical_id_int(can_id) -> int:
    s = str(can_id).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    try:
        return int(s, 16)
    except (ValueError, TypeError):
        return 0


def payload_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (bytes uint8 [N,8], valid bool [N,8]). Missing bytes -> 0/invalid."""
    n = len(df)
    out = np.zeros((n, 8), dtype=np.uint8)
    valid = np.zeros((n, 8), dtype=bool)
    dlc = df["dlc"].to_numpy(dtype=int) if "dlc" in df.columns else np.full(n, 8)
    for i, col in enumerate(BYTE_COLS):
        if col not in df.columns:
            continue
        vals = df[col].map(_hex_byte).to_numpy(dtype=np.uint8)
        present = i < dlc
        out[:, i] = vals
        valid[:, i] = present
    return out, valid


def timing_feature(ts: np.ndarray, dt_scale_ms: float) -> np.ndarray:
    ts = np.asarray(ts, dtype=float)
    dt = np.zeros_like(ts)
    if len(ts) > 1:
        dt[1:] = np.diff(ts)
    dt = np.clip(dt, 0.0, None) * 1000.0  # seconds -> ms
    feat = np.log1p(dt) / np.log1p(float(dt_scale_ms))
    return np.clip(feat, 0.0, 1.0)[:, None].astype(np.float32)


def frames_to_features(
    df: pd.DataFrame, spec: TokenizerConfig
) -> tuple[np.ndarray, dict[str, tuple[int, int]], np.ndarray]:
    """Encode a canonical frame table into ``(features[N,F], slices, byte_valid)``.

    ``df`` must be sorted by time and have ``timestamp, can_id, dlc, data_0..7``.
    """
    if df.empty:
        return (
            np.zeros((0, spec.feature_dim), dtype=np.float32),
            _slices(spec),
            np.zeros((0, 8), dtype=bool),
        )
    df = df.sort_values("timestamp").reset_index(drop=True)
    bytes_mat, byte_valid = payload_matrix(df)

    if spec.payload_mode == "byte":
        payload = bytes_mat.astype(np.float32) / 255.0
    else:
        payload = np.unpackbits(bytes_mat, axis=1).astype(np.float32)  # [N,64]

    dlc = df["dlc"].to_numpy(dtype=float) if "dlc" in df.columns else np.full(len(df), 8.0)
    dlc_feat = np.clip(dlc[:, None] / 8.0, 0.0, 1.0).astype(np.float32)

    ts = df["timestamp"].to_numpy(dtype=float)
    dt_feat = timing_feature(ts, spec.dt_scale_ms)

    parts = [payload, dlc_feat, dt_feat]
    if spec.id_mode != "none":
        ids = df["can_id"].map(_canonical_id_int).to_numpy(dtype=np.int64)
        if spec.id_mode == "bit":
            bits = np.zeros((len(df), spec.id_bits), dtype=np.float32)
            for b in range(spec.id_bits):
                bits[:, spec.id_bits - 1 - b] = (ids >> b) & 1
            parts.append(bits)
        else:
            parts.append((ids / float(2**spec.id_bits))[:, None].astype(np.float32))

    feats = np.concatenate(parts, axis=1).astype(np.float32)
    return feats, _slices(spec), byte_valid


def _slices(spec: TokenizerConfig) -> dict[str, tuple[int, int]]:
    p = spec.payload_dim
    return {
        "payload": (0, p),
        "dlc": (p, p + 1),
        "timing": (p + 1, p + 2),
        "id": (p + 2, p + 2 + spec.id_dim),
    }


def make_windows(
    features: np.ndarray, labels: np.ndarray, length: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice ``features`` into contiguous windows.

    Returns ``(X [M,L,F], y [M,L], starts [M])``. No attack filtering occurs here.
    """
    n = len(features)
    if length <= 0 or n < length:
        f = features.shape[1] if features.ndim == 2 else 0
        return (
            np.zeros((0, max(length, 0), f), dtype=np.float32),
            np.zeros((0, max(length, 0)), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    starts = np.arange(0, n - length + 1, stride, dtype=np.int64)
    X = np.stack([features[s : s + length] for s in starts]).astype(np.float32)
    y = np.stack([labels[s : s + length] for s in starts]).astype(np.int64)
    return X, y, starts


__all__ = [
    "BYTE_COLS",
    "TokenizerConfig",
    "frames_to_features",
    "make_windows",
    "payload_matrix",
    "timing_feature",
]
