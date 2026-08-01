"""Generate ROAD validation figures from results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures/road")
OUT.mkdir(parents=True, exist_ok=True)


def _grouped(results: dict) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for name, r in results.items():
        parts = name.split("_")
        atype = "_".join(parts[:-1]) if parts[-1].isdigit() else name
        g[atype].append(r)
    return g


def _agg(groups: dict[str, list[dict]], key: str) -> dict[str, float]:
    return {
        at: float(np.mean([r.get(key) for r in rs if isinstance(r.get(key), (int, float))]))
        for at, rs in groups.items()
    }


def main() -> None:
    data = json.load(open("results/road/road_results.json"))
    groups = _grouped(data)
    types = sorted(groups)
    f1 = _agg(groups, "f1")
    rec = _agg(groups, "recall")
    fpr = _agg(groups, "fpr")
    auc = _agg(groups, "roc_auc")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(len(types))
    axes[0].bar(x, [f1[t] for t in types], color="steelblue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(types, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("F1")
    axes[0].set_title("ROAD PIRD: F1 by attack type")

    axes[1].bar(x, [rec[t] for t in types], color="crimson", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(types, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Recall")

    axes[2].bar(x, [fpr[t] for t in types], color="darkorange", alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(types, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("FPR")
    axes[2].set_title("FPR")

    fig.tight_layout()
    fig.savefig(OUT / "road_pird_by_attack_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/road/road_pird_by_attack_type.png")

    # ROC-AUC summary figure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([t for t in types][::-1], [auc[t] for t in types][::-1], color="green", alpha=0.75)
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("ROAD PIRD: ROC-AUC by attack type (residual IF)")
    fig.tight_layout()
    fig.savefig(OUT / "road_pird_rocauc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/road/road_pird_rocauc.png")


if __name__ == "__main__":
    main()
