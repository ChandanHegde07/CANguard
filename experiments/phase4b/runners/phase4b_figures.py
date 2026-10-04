"""Phase 4B — scoring-method comparison and representation-shift figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.exp import save_csv, save_json  # noqa: E402
from canguard.visualization.style import (  # noqa: E402
    apply_ieee_style,
    figsize_single,
    save_ieee_figure,
)


def _bar_by_method(df, value, path, title):
    apply_ieee_style()
    zs = df[df["subset"] == "zero_shot"].copy()
    zs["method"] = np.where(
        zs["score"] == "knn", "knn(k=" + zs["k"].astype("Int64").astype(str) + ")", zs["score"]
    )
    piv = zs.pivot_table(index="method", values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=figsize_single(3.4))
    ax.bar(piv.index.astype(str), piv[value].to_numpy(dtype=float), color="#4c72b0")
    ax.set_ylabel(value)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40)
    return save_ieee_figure(fig, path)


def _shift_matrix(shift, path, title):
    apply_ieee_style()
    piv = shift.pivot_table(index="source", columns="target", values="mean_abs_smd", aggfunc="mean")
    fig, ax = plt.subplots(figsize=figsize_single(3.4))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="magma")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=6)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_ieee_figure(fig, path)


def run_figures(cfg: dict, root: Path) -> dict:
    results = root / "results"
    plots = root / "plots"
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    summary = {}
    sc = results / "score_compare" / "score_comparison.csv"
    sh = results / "score_compare" / "representation_shift.csv"
    if sc.exists():
        df = pd.read_csv(sc)
        save_csv(results / "score_comparison.csv", df)
        _bar_by_method(
            df, "tpr_at_1pct", plots / "score_tpr1.png", "Zero-shot TPR@1%FPR by scoring method"
        )
        _bar_by_method(
            df, "actual_fpr", plots / "score_fpr.png", "Zero-shot actual FPR by scoring method"
        )
        _bar_by_method(df, "f1", plots / "score_f1.png", "Zero-shot F1 by scoring method")
        summary["score_rows"] = int(len(df))
        if sh.exists():
            sdf = pd.read_csv(sh)
            save_csv(results / "representation_shift.csv", sdf)
            _shift_matrix(sdf, plots / "shift_matrix.png", "Source→target normal SMD")
            zs = sdf[sdf["subset"] == "zero_shot"]
            fig, ax = plt.subplots(figsize=figsize_single(3.4))
            apply_ieee_style()
            lab = zs["source"] + "→" + zs["target"]
            ax.barh(lab, zs["attack_normal_distance_ratio"], color="#c44e52")
            ax.axvline(1.0, color="k", lw=0.8)
            ax.set_xlabel("attack/normal distance ratio")
            ax.set_title("Representation separability")
            save_ieee_figure(fig, plots / "attack_separation.png")
            summary["shift_rows"] = int(len(sdf))
    save_json(results / "phase4b_figures.json", summary)
    print("Phase 4B figures:", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4B figures")
    parser.add_argument("--config", default="")
    parser.parse_args()
    run_figures({}, Path("experiments/phase4b"))


if __name__ == "__main__":
    main()
