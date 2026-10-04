"""Source→target normal-distribution shift diagnostics (Phase 3).

Compares source normal traffic and target normal traffic (no attack samples) to
characterise domain shift. SMD is a diagnostic effect size, not causal evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .distribution_shift import standardized_mean_difference
from .transfer_metrics import canonicalize_frame_ids

EPS = 1e-12


def feature_shift_table(
    source_norm: pd.DataFrame, target_norm: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Per-feature source-vs-target SMD (target minus source), ranked by |SMD|."""
    rows = []
    for f in feature_cols:
        s = np.asarray(source_norm[f], dtype=float)
        t = np.asarray(target_norm[f], dtype=float)
        s = s[np.isfinite(s)]
        t = t[np.isfinite(t)]
        smd = standardized_mean_difference(s, t)
        rows.append({
            "feature": f,
            "source_mean": float(np.mean(s)) if s.size else float("nan"),
            "target_mean": float(np.mean(t)) if t.size else float("nan"),
            "source_std": float(np.std(s, ddof=1)) if s.size > 1 else float("nan"),
            "target_std": float(np.std(t, ddof=1)) if t.size > 1 else float("nan"),
            "source_median": float(np.median(s)) if s.size else float("nan"),
            "target_median": float(np.median(t)) if t.size else float("nan"),
            "smd": smd,
            "abs_smd": abs(smd) if np.isfinite(smd) else float("nan"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("abs_smd", ascending=False).reset_index(drop=True)


def summarize_shift(shift_table: pd.DataFrame) -> dict:
    if shift_table.empty:
        return {"mean_abs_smd": float("nan"), "median_abs_smd": float("nan"),
                "max_abs_smd": float("nan")}
    a = shift_table["abs_smd"].to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean_abs_smd": float("nan"), "median_abs_smd": float("nan"),
                "max_abs_smd": float("nan")}
    return {
        "mean_abs_smd": float(np.mean(a)),
        "median_abs_smd": float(np.median(a)),
        "max_abs_smd": float(np.max(a)),
    }


def per_id_smd(
    source_norm: pd.DataFrame,
    target_norm: pd.DataFrame,
    feature_cols: list[str],
    min_windows: int = 20,
) -> pd.DataFrame:
    """Per-ID mean |SMD| over features for IDs present in both domains."""
    s = source_norm.copy()
    t = target_norm.copy()
    s["can_id"] = canonicalize_frame_ids(s["can_id"])
    t["can_id"] = canonicalize_frame_ids(t["can_id"])
    common = sorted(set(s["can_id"]) & set(t["can_id"]))
    rows = []
    for cid in common:
        sg = s[s["can_id"] == cid]
        tg = t[t["can_id"] == cid]
        if len(sg) < min_windows or len(tg) < min_windows:
            continue
        smds = []
        for f in feature_cols:
            v = standardized_mean_difference(sg[f], tg[f])
            if np.isfinite(v):
                smds.append(abs(v))
        if smds:
            rows.append({"can_id": cid, "n_source": len(sg), "n_target": len(tg),
                         "mean_abs_smd": float(np.mean(smds)),
                         "max_abs_smd": float(np.max(smds))})
    if not rows:
        return pd.DataFrame(
            columns=["can_id", "n_source", "n_target", "mean_abs_smd", "max_abs_smd"]
        )
    return pd.DataFrame(rows).sort_values("mean_abs_smd", ascending=False).reset_index(drop=True)


__all__ = ["feature_shift_table", "per_id_smd", "summarize_shift"]
