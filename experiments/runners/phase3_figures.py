"""Phase 3 — aggregate transfer results, build the transfer matrix and figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.exp import load_config, save_csv, save_json  # noqa: E402
from canguard.visualization.transfer import (  # noqa: E402
    plot_f1_bars,
    plot_id_overlap,
    plot_operating_curve,
    plot_smd,
    plot_transfer_heatmap,
)

TRANSFER_DIRS = ["hcrl_to_road", "road_to_hcrl"]


def _load_summaries(root: Path) -> pd.DataFrame:
    frames = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        s = d / "summary.csv"
        if s.exists():
            frames.append(pd.read_csv(s))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_figures(cfg: dict, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    plots = root / "plots"
    results = root / "results"
    plots.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    matrix = _load_summaries(root)
    if not matrix.empty:
        save_csv(results / "transfer_matrix.csv", matrix)
        matrix["transfer"] = matrix["source_dataset"] + "→" + matrix["target_dataset"]
        plot_transfer_heatmap(matrix, "primary_recall", plots / "transfer_recall.png",
                              title="Zero-shot recall (known-ID)")
        plot_transfer_heatmap(matrix, "primary_f1", plots / "transfer_f1.png",
                              title="Zero-shot F1 (known-ID)")
        plot_transfer_heatmap(matrix, "target_actual_fpr", plots / "transfer_fpr.png",
                              title="Actual target FPR (known-ID)", cmap="magma")
        plot_id_overlap(matrix, "transfer", plots / "id_overlap.png",
                        title="CAN-ID overlap (source vs target)")
        plot_f1_bars(matrix, "transfer", plots / "transfer_f1_bars.png",
                     title="PIRD known-ID vs unseen-ID fallback F1")
        for _, r in matrix.iterrows():
            d = root / f"{r['source_dataset']}_to_{r['target_dataset']}"
            sf = d / "feature_shift.csv"
            if sf.exists():
                fname = f"smd_{r['source_dataset']}_to_{r['target_dataset']}.png"
                plot_smd(pd.read_csv(sf), plots / fname,
                         title=f"SMD {r['source_dataset']}→{r['target_dataset']} (normals)")
            ff = d / "fixed_fpr.csv"
            if ff.exists():
                oname = f"operating_{r['source_dataset']}_to_{r['target_dataset']}.png"
                plot_operating_curve(
                    pd.read_csv(ff), plots / oname,
                    title=f"Recall vs target FPR "
                          f"({r['source_dataset']}→{r['target_dataset']})")

    for sub, outname, title in (
        ("cross_capture", "capture_transfer_f1.png", "ROAD cross-capture F1 (known-ID)"),
        ("gem_captures", "gem_capture_transfer_f1.png", "GEM-CAN cross-capture F1 (known-ID)"),
    ):
        cap = root / sub / "capture_transfer.csv"
        if not cap.exists():
            continue
        cdf = pd.read_csv(cap)
        save_csv(results / f"{sub}_transfer.csv", cdf)
        # capture_transfer.csv already carries dataset columns; use the capture
        # names as the heatmap axes without creating duplicate columns.
        heat = cdf.drop(columns=[c for c in ("source_dataset", "target_dataset")
                                 if c in cdf.columns]).copy()
        heat["source_dataset"] = cdf["source_capture"]
        heat["target_dataset"] = cdf["target_capture"]
        plot_transfer_heatmap(heat, "primary_f1", plots / outname, title=title)

    # Cross-vehicle transfer matrix (source vehicle x target vehicle).
    veh = root / "survival_vehicle" / "vehicle_transfer.csv"
    if veh.exists():
        vdf = pd.read_csv(veh)
        save_csv(results / "vehicle_transfer.csv", vdf)
        vf = vdf.drop(columns=[c for c in ("source_dataset", "target_dataset")
                               if c in vdf.columns]).copy()
        vf["source_dataset"] = vdf["source_vehicle"]
        vf["target_dataset"] = vdf["target_vehicle"]
        plot_transfer_heatmap(vf, "primary_f1", plots / "vehicle_transfer_f1.png",
                              title="Cross-vehicle F1 (known-ID)")
        plot_transfer_heatmap(vf, "primary_recall", plots / "vehicle_transfer_recall.png",
                              title="Cross-vehicle recall (known-ID)")
        plot_transfer_heatmap(vf, "target_actual_fpr", plots / "vehicle_transfer_fpr.png",
                              title="Cross-vehicle actual FPR (known-ID)", cmap="magma")
        plot_id_overlap(vf, "source_dataset", plots / "vehicle_id_overlap.png",
                        title="CAN-ID overlap (source vs target vehicle)")

    summary = {
        "transfer_matrix_rows": int(len(matrix)),
        "transfers": matrix["transfer"].tolist() if not matrix.empty else [],
    }
    save_json(results / "phase3_summary.json", summary)
    # Windows consoles default to cp1252; keep stdout ASCII-safe.
    print("Phase 3 figures:", str(summary).replace("\u2192", "->"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 figures/matrix")
    parser.add_argument("--config", default="experiments/configs/phase3_figures.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config) if Path(args.config).exists() else {}
    root = Path(cfg.get("output_dir", "experiments/phase3"))
    run_figures(cfg, root)


if __name__ == "__main__":
    main()
