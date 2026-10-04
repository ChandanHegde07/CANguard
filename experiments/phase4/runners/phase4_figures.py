"""Phase 4 — aggregate zero-shot/ablation results into transfer matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
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


def _matrix(df: pd.DataFrame, value: str, path: Path, title: str) -> None:
    apply_ieee_style()
    piv = df.pivot_table(index="source", columns="target", values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=figsize_single(3.4))
    im = ax.imshow(piv.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=6)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=6)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.5 else "black")
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_ieee_figure(fig, path)


def run_figures(cfg: dict, root: Path) -> dict:
    results = root / "results"
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    summary = {}
    zs = results / "zero_shot" / "zero_shot_results.csv"
    if zs.exists():
        df = pd.read_csv(zs)
        save_csv(results / "zero_shot_matrix.csv", df)
        _matrix(df, "f1", plots / "zero_shot_f1.png", "CAN-TFM zero-shot F1")
        _matrix(df, "actual_fpr", plots / "zero_shot_fpr.png", "CAN-TFM zero-shot FPR")
        _matrix(df, "tpr_at_1pct_fpr", plots / "zero_shot_tpr1.png", "CAN-TFM TPR@1%FPR")
        summary["zero_shot_cells"] = int(len(df))
    cid = results / "cross_id" / "cross_id_results.csv"
    if cid.exists():
        df = pd.read_csv(cid)
        save_csv(results / "cross_id.csv", df)
        un = df[df["subset"] == "unseen"]
        if not un.empty:
            _matrix(un, "tpr_at_1pct_fpr", plots / "cross_id_unseen_tpr1.png",
                    "Cross-ID (unseen) TPR@1%FPR")
        summary["cross_id_rows"] = int(len(df))
    save_json(results / "phase4_summary.json", summary)
    print("Phase 4 figures:", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 figures")
    parser.add_argument("--config", default="experiments/phase4/configs/phase4_figures.yaml")
    parser.parse_args()
    root = Path("experiments/phase4")
    run_figures({}, root)


if __name__ == "__main__":
    main()
