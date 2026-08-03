"""Detection latency: attack start → first true positive."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detection_latency(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict:
    """Compute latency for the first attack segment in a test sequence.

    Parameters
    ----------
    timestamps
        Window timestamps (seconds), chronological.
    y_true, y_pred
        Binary labels/predictions aligned with timestamps.

    Returns
    -------
    dict with delay_frames, delay_ms, detected, attack_start_idx, detect_idx, ...
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    timestamps = np.asarray(timestamps, dtype=float)

    attack_idx = np.where(y_true == 1)[0]
    if len(attack_idx) == 0:
        return {
            "detected": False,
            "delay_frames": float("nan"),
            "delay_ms": float("nan"),
            "attack_start_idx": -1,
            "detect_idx": -1,
            "n_attack_windows": 0,
        }

    # First contiguous attack segment
    start = int(attack_idx[0])
    # extend while consecutive
    end = start
    while end + 1 < len(y_true) and y_true[end + 1] == 1:
        end += 1

    # first TP at or after start within this segment or any later attack
    post = np.where((np.arange(len(y_pred)) >= start) & (y_pred == 1) & (y_true == 1))[0]
    if len(post) == 0:
        return {
            "detected": False,
            "delay_frames": float("nan"),
            "delay_ms": float("nan"),
            "attack_start_idx": start,
            "detect_idx": -1,
            "n_attack_windows": int((y_true == 1).sum()),
            "missed": True,
        }

    det = int(post[0])
    delay_frames = det - start
    delay_ms = float((timestamps[det] - timestamps[start]) * 1000.0)
    return {
        "detected": True,
        "delay_frames": int(delay_frames),
        "delay_ms": delay_ms,
        "attack_start_idx": start,
        "detect_idx": det,
        "n_attack_windows": int((y_true == 1).sum()),
        "missed": False,
    }


def multi_segment_latencies(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Latency for each contiguous attack segment."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    timestamps = np.asarray(timestamps, dtype=float)

    rows = []
    i = 0
    seg = 0
    n = len(y_true)
    while i < n:
        if y_true[i] != 1:
            i += 1
            continue
        start = i
        while i < n and y_true[i] == 1:
            i += 1
        end = i - 1
        # first TP in [start, end]
        hits = [j for j in range(start, end + 1) if y_pred[j] == 1]
        if hits:
            det = hits[0]
            rows.append(
                {
                    "segment": seg,
                    "detected": True,
                    "delay_frames": det - start,
                    "delay_ms": float((timestamps[det] - timestamps[start]) * 1000.0),
                    "segment_len": end - start + 1,
                }
            )
        else:
            rows.append(
                {
                    "segment": seg,
                    "detected": False,
                    "delay_frames": float("nan"),
                    "delay_ms": float("nan"),
                    "segment_len": end - start + 1,
                }
            )
        seg += 1
    return pd.DataFrame(rows)
