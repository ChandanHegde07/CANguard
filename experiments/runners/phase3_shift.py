"""Phase 3 — consolidated source→target distribution-shift diagnostic.

PIRD's per-ID statistics assume that a CAN ID behaves similarly in source and
target. This runner collects the per-transfer shift tables produced by
``run_phase3_transfer`` (``feature_shift.csv``, ``per_id_shift.csv``) and
aggregates them into a single machine-readable view:

    experiments/phase3/distribution_shift/
        all_feature_shift.csv     per (transfer, feature) SMD
        all_per_id_shift.csv      per (transfer, CAN ID) mean |SMD|
        shift_summary.csv         per-transfer mean/median/max |SMD|
        domain_profile.csv        per-domain feature location/scale + ID census
        shift_summary.json

plus ``plots/smd_<transfer>.png`` and ``plots/shift_means.png``.

SMD is a diagnostic *effect size*, not causal evidence. Attack samples are never
used here: only source normals vs target normals are compared.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.exp import load_config, save_csv, save_json  # noqa: E402
from canguard.visualization.transfer import plot_smd  # noqa: E402


def _iter_transfer_dirs(root: Path):
    """Yield every directory holding a per-transfer summary/shift table.

    Recurses so that per-pair cross-vehicle subdirectories (under
    ``survival_vehicle/<source>_to_<target>/``) are included alongside the
    top-level dataset-transfer directories.
    """
    seen = set()
    for name in ("summary.csv", "feature_shift.csv"):
        for path in sorted(root.rglob(name)):
            d = path.parent
            if d.name in {"dataset_audit", "distribution_shift", "results", "plots", "cache"}:
                continue
            if d in seen:
                continue
            seen.add(d)
            yield d


def _domains_from_summary(row: dict, dname: str) -> tuple[str, str]:
    """Human-readable source/target domain labels for a transfer row."""
    sv = row.get("source_vehicle")
    tv = row.get("target_vehicle")
    if isinstance(sv, str) and sv.strip():
        return f"survival:{sv}", f"survival:{tv}"
    return (str(row.get("source_dataset", dname)), str(row.get("target_dataset", dname)))


def _safe_token(name: str) -> str:
    """Filesystem-safe token (Windows forbids ``:`` and friends)."""
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(name))


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty frame for missing/zero-byte files."""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def _tag(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "target_domain", target)
    df.insert(0, "source_domain", source)
    return df


def run_shift(cfg: dict, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    out = root / "distribution_shift"
    plots = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    feature_rows: list[pd.DataFrame] = []
    id_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for d in _iter_transfer_dirs(root):
        s = d / "summary.csv"
        src = tgt = d.name
        if s.exists():
            sdf = _read_csv_safe(s)
            if sdf.empty:
                continue
            row = sdf.iloc[0].to_dict()
            src, tgt = _domains_from_summary(row, d.name)
            summary_rows.append({
                "source_domain": src,
                "target_domain": tgt,
                "source_vehicle": row.get("source_vehicle", ""),
                "target_vehicle": row.get("target_vehicle", ""),
                "transfer_dir": str(d.relative_to(root)),
                "n_shared_ids": row.get("n_shared_ids"),
                "n_target_ids": row.get("n_target_ids"),
                "n_target_only_ids": row.get("n_target_only_ids"),
                "frame_weighted_overlap": row.get("frame_weighted_overlap"),
                "target_unseen_id_window_fraction": row.get("target_unseen_id_window_fraction"),
                "target_actual_fpr": row.get("target_actual_fpr"),
                "primary_recall": row.get("primary_recall"),
                "primary_f1": row.get("primary_f1"),
                "mean_abs_smd": row.get("mean_abs_smd"),
                "median_abs_smd": row.get("median_abs_smd"),
                "max_abs_smd": row.get("max_abs_smd"),
            })
        fs = d / "feature_shift.csv"
        if fs.exists():
            df = _read_csv_safe(fs)
            if not df.empty:
                feature_rows.append(_tag(df, src, tgt))
        ps = d / "per_id_shift.csv"
        if ps.exists():
            df = _read_csv_safe(ps)
            if not df.empty:
                id_rows.append(_tag(df, src, tgt))

    all_features = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
    all_ids = pd.concat(id_rows, ignore_index=True) if id_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    save_csv(out / "all_feature_shift.csv", all_features)
    save_csv(out / "all_per_id_shift.csv", all_ids)
    save_csv(out / "shift_summary.csv", summary)

    # Per-domain feature profile (location/scale) from the consolidated table.
    profile = _domain_profile(all_features)
    save_csv(out / "domain_profile.csv", profile)

    # Per-transfer SMD bars (sanitise filenames for Windows).
    for (src, tgt), g in all_features.groupby(["source_domain", "target_domain"]):
        if g.empty:
            continue
        plot_smd(g, plots / f"smd_{_safe_token(src)}_to_{_safe_token(tgt)}.png",
                 title=f"SMD {src}→{tgt} normals")
    if not all_ids.empty:
        from canguard.visualization.transfer import plot_per_id_smd

        plot_per_id_smd(all_ids, plots / "per_id_smd.png",
                        title="Per-ID normal shift (shared IDs)")
    _plot_shift_means(summary, plots / "shift_means.png")

    payload = {
        "n_transfers": int(len(summary)),
        "transfers": summary[["source_domain", "target_domain"]].to_dict("records")
        if not summary.empty else [],
        "mean_abs_smd_by_transfer": summary.set_index(
            summary["source_domain"] + "→" + summary["target_domain"]
        )["mean_abs_smd"].to_dict() if not summary.empty else {},
    }
    save_json(out / "shift_summary.json", payload)
    print("Phase 3 shift:", payload["n_transfers"], "transfers")
    return {"summary": summary, "features": all_features, "ids": all_ids,
            "profile": profile}


def _domain_profile(all_features: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-transfer feature shift into a compact domain profile."""
    if all_features.empty:
        return pd.DataFrame(
            columns=["source_domain", "target_domain", "n_features",
                     "mean_abs_smd", "median_abs_smd", "max_abs_smd",
                     "most_shifted_feature", "most_shifted_smd"]
        )
    rows = []
    for (src, tgt), g in all_features.groupby(["source_domain", "target_domain"]):
        a = g["abs_smd"].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        top = g.sort_values("abs_smd", ascending=False).iloc[0] if len(g) else None
        rows.append({
            "source_domain": src, "target_domain": tgt,
            "n_features": int(len(g)),
            "mean_abs_smd": float(np.mean(a)) if a.size else float("nan"),
            "median_abs_smd": float(np.median(a)) if a.size else float("nan"),
            "max_abs_smd": float(np.max(a)) if a.size else float("nan"),
            "most_shifted_feature": top["feature"] if top is not None else "",
            "most_shifted_smd": float(top["smd"]) if top is not None else float("nan"),
        })
    return pd.DataFrame(rows)


def _plot_shift_means(summary: pd.DataFrame, path: Path) -> Path | None:
    if summary.empty or "mean_abs_smd" not in summary.columns:
        return None
    import matplotlib.pyplot as plt

    from canguard.visualization.style import (
        IEEE_COLORS,
        apply_ieee_style,
        figsize_single,
        save_ieee_figure,
    )

    apply_ieee_style()
    d = summary.dropna(subset=["mean_abs_smd"]).copy()
    if d.empty:
        return None
    d["label"] = d["source_domain"].astype(str) + "→" + d["target_domain"].astype(str)
    fig, ax = plt.subplots(figsize=figsize_single(3.2))
    ax.barh(d["label"], d["mean_abs_smd"], color=IEEE_COLORS["residual"])
    ax.set_xlabel("mean |SMD| (normals)")
    ax.set_title("Source→target normal shift")
    return save_ieee_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 distribution-shift aggregation")
    parser.add_argument("--config", default="experiments/configs/phase3_shift.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    root = Path(cfg.get("output_dir", "experiments/phase3"))
    run_shift(cfg, root)
    print(f"Shift diagnostics written to {root / 'distribution_shift'}")


if __name__ == "__main__":
    main()
