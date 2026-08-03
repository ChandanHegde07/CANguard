"""Manuscript-facing table formatters and ROAD summary generation.

Rules:
  * ROAD numbers only from corrected pre_injection_v1 tables.
  * Never print mean ± std when n_captures == 1 (use n/a (n=1)).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metadata import (
    PHASE_WORKSHOP,
    PROTOCOL_ROAD_PRE_INJECTION,
    assert_road_protocol_ok,
    ensure_protocol_column,
    tag_dataframe,
)


def format_mean_std(mean: float, std: float | None, n: int, decimals: int = 3) -> str:
    """Format mean ± std for manuscript; suppress std when n < 2."""
    if mean != mean:  # NaN
        return "n/a"
    m = f"{float(mean):.{decimals}f}"
    if n is None or int(n) < 2:
        return f"{m} (n={int(n) if n is not None else 1}; std n/a)"
    if std is None or std != std:
        return f"{m} (n={int(n)}; std n/a)"
    return f"{m} ± {float(std):.{decimals}f} (n={int(n)})"


def build_road_attack_summary(
    road_results: pd.DataFrame,
    *,
    representation: str = "residual",
    metric_cols: tuple[str, ...] = ("f1", "recall", "precision", "roc_auc", "pr_auc", "fpr"),
) -> pd.DataFrame:
    """Aggregate ROAD per-attack-type metrics with n_captures; corrected protocol only."""
    df = ensure_protocol_column(road_results)
    assert_road_protocol_ok(df)

    if "representation" in df.columns:
        df = df[df["representation"] == representation].copy()

    # One row per capture×detector; n_captures is unique captures per attack_type
    group_keys = ["attack_type", "detector"]
    if "representation" in df.columns:
        group_keys = ["attack_type", "detector", "representation"]

    rows = []
    for keys, g in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_keys, keys))
        n_cap = int(g["capture"].nunique()) if "capture" in g.columns else int(len(g))
        rec["n_captures"] = n_cap
        for m in metric_cols:
            if m not in g.columns:
                continue
            vals = g[m].astype(float)
            rec[f"{m}_mean"] = float(vals.mean())
            rec[f"{m}_std"] = float(vals.std(ddof=1)) if n_cap >= 2 and len(vals) >= 2 else float("nan")
            rec[f"{m}_fmt"] = format_mean_std(
                rec[f"{m}_mean"],
                rec[f"{m}_std"] if n_cap >= 2 else None,
                n_cap,
            )
        rows.append(rec)

    out = pd.DataFrame(rows)
    out = tag_dataframe(
        out,
        protocol_version=PROTOCOL_ROAD_PRE_INJECTION,
        phase=PHASE_WORKSHOP,
        extra={"source_table": "road_representation_results.csv"},
    )
    return out


def write_road_manuscript_summary(
    road_results_path: str | Path,
    out_path: str | Path,
    *,
    representation: str = "residual",
) -> pd.DataFrame:
    """Load ROAD results, assert protocol, write manuscript summary CSV."""
    path = Path(road_results_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing ROAD results: {path}")
    df = pd.read_csv(path)
    summary = build_road_attack_summary(df, representation=representation)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    return summary
