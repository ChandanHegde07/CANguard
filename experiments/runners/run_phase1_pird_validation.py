"""Phase 1 — PIRD validation framework.

This phase interrogates whether *Per-ID Behavioral Residual Detection (PIRD)*
is genuinely responsible for detection quality, and whether it is deployable.
Four experiment groups map directly onto the validation plan:

1. ``cross_condition``    — cross-capture / cross-condition generalization. Fit
   PIRD on one driving condition's normals, freeze it, evaluate on a different
   condition's stream (no target data in fitting or thresholding).
2. ``global_threshold``   — deployment realism. Compare a single pooled decision
   threshold against per-dataset thresholds, measuring how consistently one
   fixed operating point transfers across conditions.
3. ``residualization``    — representation study. Compare raw features, global
   standardization, per-ID z-score, robust per-ID MAD, and per-ID quantile
   normalization to isolate the contribution of residualization.
4. ``window_sensitivity`` — window-size sweep reporting the detection/latency
   trade-off (score cost per window + detection delay).

Every group reports deployment-oriented metrics alongside F1: Recall@fixed FPR,
FPR@target recall, false alarms per hour, and detection latency.

Usage:
    python -m experiments.runners.run_phase1_pird_validation \
        --config experiments/configs/phase1_pird_validation.yaml [--mode all]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.detectors import create_detector  # noqa: E402
from canguard.evaluation import (  # noqa: E402
    compute_metrics,
    deployment_report,
    flatten_deployment,
)
from canguard.evaluation.threshold import choose_threshold_from_val_normals  # noqa: E402
from canguard.exp import (  # noqa: E402
    FeatureCache,
    load_config,
    save_csv,
    save_json,
    set_global_seed,
    timed,
)
from canguard.exp.matrix import build_window_table, resolve_data_path  # noqa: E402
from canguard.features import BEHAVIORAL_FEATURES_V1, temporal_split  # noqa: E402
from canguard.transforms import build_residualizer  # noqa: E402

logger = logging.getLogger("canguard")

DEFAULT_SPLIT = {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4}
DEFAULT_STRATEGIES = ["raw", "global_z", "per_id_z", "per_id_mad", "per_id_quantile"]
METRIC_KEYS = ("precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "tp", "fp", "fn", "tn")


def _feature_cols(cfg: dict) -> list[str]:
    return list(cfg.get("feature_cols") or BEHAVIORAL_FEATURES_V1)


def _make_detector(cfg: dict):
    d = dict(cfg.get("detector", {}))
    kind = d.pop("kind", "isolation_forest")
    return create_detector(kind, **d)


def _eval_cfg(cfg: dict) -> dict:
    return cfg.get("evaluation", {})


def _build_windows(cfg: dict, name: str, window_size: int, cache: FeatureCache) -> pd.DataFrame:
    path = resolve_data_path(cfg.get("data_dir", "data"), name)
    return build_window_table(
        path,
        window_size=window_size,
        sample_size=cfg.get("sample_size"),
        cache=cache,
    )


def _prepare(
    cfg: dict,
    name: str,
    window_size: int,
    residualizer,
    feature_cols: list[str],
    cache: FeatureCache,
) -> dict:
    """Fit a residualizer + detector on one dataset and score its test split.

    Reference statistics come from the calibration split's normals; the detector
    is fit on train normals with a validation holdout for threshold selection;
    the test split is never used for fitting.
    """
    split = cfg.get("split", DEFAULT_SPLIT)
    wt = _build_windows(cfg, name, window_size, cache)
    calib, train, test = temporal_split(
        wt, split["calib_frac"], split["train_frac"], split["test_frac"]
    )
    residualizer.fit(calib, feature_cols)
    res_cols = [c + "_res" for c in feature_cols]
    res_train = residualizer.transform(train, feature_cols)
    res_test = residualizer.transform(test, feature_cols)

    train_norm = res_train[res_train["is_attack"] == 0]
    n_val = max(1, int(len(train_norm) * float(_eval_cfg(cfg).get("val_holdout_fraction", 0.2))))
    if n_val < len(train_norm):
        fit_df = train_norm.iloc[:-n_val]
        val_df = train_norm.iloc[-n_val:]
    else:
        fit_df = train_norm
        val_df = train_norm

    detector = _make_detector(cfg)
    with timed() as t_fit:
        detector.fit(fit_df[res_cols].fillna(0).values)
    with timed() as t_score:
        val_scores = detector.score_samples(val_df[res_cols].fillna(0).values)
        test_scores = detector.score_samples(res_test[res_cols].fillna(0).values)

    return {
        "detector": detector,
        "res_cols": res_cols,
        "test": test,
        "val_scores": np.asarray(val_scores, dtype=float),
        "test_scores": np.asarray(test_scores, dtype=float),
        "y_test": test["is_attack"].to_numpy(dtype=int),
        "timestamps": test["timestamp"].to_numpy(dtype=float),
        "fit_seconds": t_fit["seconds"],
        "score_seconds": t_score["seconds"],
        "n_train_normals": int(len(fit_df)),
        "n_val_normals": int(len(val_df)),
        "n_test_windows": int(len(test)),
        "n_known_ids": len(getattr(residualizer, "stats_", {}) or {}),
    }


def _metric_row(p: dict, threshold: float, cfg: dict, **extra) -> dict:
    scores = p["test_scores"]
    y = p["y_test"]
    y_pred = (scores >= float(threshold)).astype(int)
    m = compute_metrics(y, y_pred, scores)
    dep = deployment_report(
        p["val_scores"],
        p["timestamps"],
        y,
        scores,
        y_pred,
        fpr_targets=tuple(_eval_cfg(cfg).get("fixed_fpr_targets", [0.001, 0.01, 0.05])),
        recall_target=float(_eval_cfg(cfg).get("recall_target", 0.95)),
    )
    row = dict(extra)
    row.update({k: m[k] for k in METRIC_KEYS})
    row["threshold"] = float(threshold)
    row["n_train_normals"] = p["n_train_normals"]
    row["n_val_normals"] = p["n_val_normals"]
    row["n_test_windows"] = p["n_test_windows"]
    row["fit_seconds"] = p["fit_seconds"]
    row["score_seconds"] = p["score_seconds"]
    row["ms_per_window"] = (
        p["score_seconds"] / p["n_test_windows"] * 1000.0 if p["n_test_windows"] else float("nan")
    )
    row.update(flatten_deployment(dep))
    return row


def _per_dataset_threshold(p: dict, cfg: dict) -> float:
    return choose_threshold_from_val_normals(
        p["val_scores"], float(_eval_cfg(cfg).get("fpr_target", 0.01))
    )


# ---------------------------------------------------------------------------
# Experiment 1 — cross-condition / cross-capture generalization
# ---------------------------------------------------------------------------
def run_cross_condition(cfg: dict, out_root: Path) -> pd.DataFrame:
    feature_cols = _feature_cols(cfg)
    window_size = int(cfg.get("window_size", 30))
    strategy = str(cfg.get("strategy", "per_id_z"))
    split = cfg.get("split", DEFAULT_SPLIT)
    fpr = float(_eval_cfg(cfg).get("fpr_target", 0.01))
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase1"))

    # Build each condition's window table exactly once.
    tables: dict[str, pd.DataFrame] = {}
    for name in cfg["datasets"]:
        try:
            tables[name] = _build_windows(cfg, name, window_size, cache)
        except FileNotFoundError as exc:
            logger.warning("skipping %s: %s", name, exc)

    rows: list[dict] = []
    for src_name, src_wt in tables.items():
        calib, train, _ = temporal_split(
            src_wt, split["calib_frac"], split["train_frac"], split["test_frac"]
        )
        calib_norm = calib[calib["is_attack"] == 0]
        train_norm = train[train["is_attack"] == 0]
        if len(calib_norm) < 20 or len(train_norm) < 10:
            logger.warning("skipping source %s: insufficient normals", src_name)
            continue

        residualizer = build_residualizer(strategy)
        residualizer.fit(calib_norm, feature_cols)
        res_cols = [c + "_res" for c in feature_cols]
        res_train = residualizer.transform(train_norm, feature_cols)
        n_val = max(1, int(len(res_train) * 0.2))
        fit_df = res_train.iloc[:-n_val] if n_val < len(res_train) else res_train
        val_df = res_train.iloc[-n_val:] if n_val < len(res_train) else res_train

        detector = _make_detector(cfg)
        detector.fit(fit_df[res_cols].fillna(0).values)
        val_scores = detector.score_samples(val_df[res_cols].fillna(0).values)
        threshold = choose_threshold_from_val_normals(val_scores, fpr)
        source_ids = set(calib_norm["can_id"].astype(str).unique())

        for tgt_name, tgt_wt in tables.items():
            if tgt_name == src_name:
                continue
            _, _, target = temporal_split(
                tgt_wt, split["calib_frac"], split["train_frac"], split["test_frac"]
            )
            if target.empty:
                continue
            res_target = residualizer.transform(target, feature_cols)
            with timed() as t_score:
                scores = detector.score_samples(res_target[res_cols].fillna(0).values)
            y = target["is_attack"].to_numpy(dtype=int)
            y_pred = (scores >= float(threshold)).astype(int)
            dep = deployment_report(
                val_scores,
                target["timestamp"].to_numpy(dtype=float),
                y,
                scores,
                y_pred,
                fpr_targets=tuple(_eval_cfg(cfg).get("fixed_fpr_targets", [0.001, 0.01, 0.05])),
                recall_target=float(_eval_cfg(cfg).get("recall_target", 0.95)),
            )
            m = compute_metrics(y, y_pred, scores)
            tgt_ids = set(target["can_id"].astype(str).unique())
            shared = source_ids & tgt_ids
            row = {
                "representation": strategy,
                "source_condition": src_name,
                "target_condition": tgt_name,
                "n_source_ids": len(source_ids),
                "n_target_ids": len(tgt_ids),
                "id_count_overlap": len(shared) / len(tgt_ids) if tgt_ids else float("nan"),
                "n_test_windows": int(len(target)),
                "threshold": float(threshold),
                "source_threshold": float(threshold),
                "score_seconds": t_score["seconds"],
            }
            row.update({k: m[k] for k in METRIC_KEYS})
            row.update(flatten_deployment(dep))
            rows.append(row)

    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    save_csv(out_root / "cross_condition_results.csv", df)
    return df


# ---------------------------------------------------------------------------
# Experiment 2 — global vs per-dataset threshold
# ---------------------------------------------------------------------------
def run_global_threshold(cfg: dict, out_root: Path) -> pd.DataFrame:
    feature_cols = _feature_cols(cfg)
    window_size = int(cfg.get("window_size", 30))
    strategies = cfg.get("threshold_strategies") or ["per_id_z"]
    fpr = float(_eval_cfg(cfg).get("fpr_target", 0.01))
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase1"))
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    thresholds: dict[str, float] = {}
    for strategy in strategies:
        preps: dict[str, dict] = {}
        for name in cfg["datasets"]:
            try:
                preps[name] = _prepare(
                    cfg, name, window_size, build_residualizer(strategy), feature_cols, cache
                )
            except FileNotFoundError as exc:
                logger.warning("skipping %s: %s", name, exc)
        if not preps:
            continue

        pooled = np.concatenate([p["val_scores"] for p in preps.values()])
        global_threshold = float(np.percentile(pooled, (1.0 - fpr) * 100.0))
        thresholds[strategy] = global_threshold
        per_ds = {name: _per_dataset_threshold(p, cfg) for name, p in preps.items()}

        for name, p in preps.items():
            rows.append(
                _metric_row(
                    p, per_ds[name], cfg,
                    representation=strategy, dataset=name, threshold_mode="per_dataset",
                    global_threshold=global_threshold,
                )
            )
            rows.append(
                _metric_row(
                    p, global_threshold, cfg,
                    representation=strategy, dataset=name, threshold_mode="global",
                    global_threshold=global_threshold,
                )
            )

    df = pd.DataFrame(rows)
    save_csv(out_root / "global_threshold_results.csv", df)

    # Consistency summary: how stable is each threshold mode across datasets?
    summary_rows = []
    if not df.empty:
        for (strategy, mode), g in df.groupby(["representation", "threshold_mode"]):
            summary_rows.append({
                "representation": strategy,
                "threshold_mode": mode,
                "n_datasets": len(g),
                "mean_recall": float(g["recall"].mean()),
                "std_recall": float(g["recall"].std(ddof=0)),
                "mean_fpr": float(g["fpr"].mean()),
                "std_fpr": float(g["fpr"].std(ddof=0)),
                "mean_f1": float(g["f1"].mean()),
                "mean_false_alarms_per_hour": float(g["false_alarms_per_hour"].mean()),
            })
    save_csv(out_root / "global_threshold_summary.csv", pd.DataFrame(summary_rows))
    if thresholds:
        save_json(out_root / "global_thresholds.json", thresholds)
    return df


# ---------------------------------------------------------------------------
# Experiment 3 — residualization strategy comparison
# ---------------------------------------------------------------------------
def run_residualization(cfg: dict, out_root: Path) -> pd.DataFrame:
    feature_cols = _feature_cols(cfg)
    window_size = int(cfg.get("window_size", 30))
    strategies = cfg.get("strategies") or DEFAULT_STRATEGIES
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase1"))
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name in cfg["datasets"]:
        for strategy in strategies:
            try:
                p = _prepare(
                    cfg, name, window_size, build_residualizer(strategy), feature_cols, cache
                )
            except FileNotFoundError as exc:
                logger.warning("skipping %s/%s: %s", name, strategy, exc)
                continue
            threshold = _per_dataset_threshold(p, cfg)
            rows.append(_metric_row(p, threshold, cfg, representation=strategy, dataset=name))
            logger.info(
                "[residualization] %s | %s | F1=%.3f recall=%.3f FPR=%.4f",
                name, strategy, rows[-1]["f1"], rows[-1]["recall"], rows[-1]["fpr"],
            )

    df = pd.DataFrame(rows)
    save_csv(out_root / "residualization_results.csv", df)

    # Deltas vs raw features, per dataset: does residualization drive the gain?
    delta_rows = []
    if not df.empty and "raw" in set(df["representation"]):
        for name, g in df.groupby("dataset"):
            raw = g[g["representation"] == "raw"]
            if raw.empty:
                continue
            raw_f1 = float(raw["f1"].iloc[0])
            raw_roc = float(raw["roc_auc"].iloc[0])
            for _, r in g.iterrows():
                delta_rows.append({
                    "dataset": name,
                    "representation": r["representation"],
                    "f1": r["f1"],
                    "delta_f1_vs_raw": r["f1"] - raw_f1,
                    "roc_auc": r["roc_auc"],
                    "delta_roc_auc_vs_raw": r["roc_auc"] - raw_roc,
                })
    save_csv(out_root / "residualization_delta_vs_raw.csv", pd.DataFrame(delta_rows))
    return df


# ---------------------------------------------------------------------------
# Experiment 4 — window-size sensitivity and latency trade-off
# ---------------------------------------------------------------------------
def run_window_sensitivity(cfg: dict, out_root: Path) -> pd.DataFrame:
    feature_cols = _feature_cols(cfg)
    window_sizes = cfg.get("window_sizes") or [10, 20, 30, 50, 100]
    strategy = str(cfg.get("strategy", "per_id_z"))
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase1"))
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for window_size in window_sizes:
        for name in cfg["datasets"]:
            try:
                p = _prepare(
                    cfg, name, int(window_size), build_residualizer(strategy), feature_cols, cache
                )
            except FileNotFoundError as exc:
                logger.warning("skipping ws=%s/%s: %s", window_size, name, exc)
                continue
            threshold = _per_dataset_threshold(p, cfg)
            row = _metric_row(
                p, threshold, cfg,
                representation=strategy, dataset=name, window_size=int(window_size),
            )
            rows.append(row)
            logger.info(
                "[window] ws=%s | %s | F1=%.3f FPR=%.4f ms/win=%.3f delay=%.1fms",
                window_size, name, row["f1"], row["fpr"],
                row["ms_per_window"], row["mean_delay_ms"],
            )

    df = pd.DataFrame(rows)
    save_csv(out_root / "window_sensitivity_results.csv", df)

    # Aggregate trade-off across datasets per window size.
    tradeoff_rows = []
    if not df.empty:
        for ws, g in df.groupby("window_size"):
            tradeoff_rows.append({
                "window_size": int(ws),
                "mean_f1": float(g["f1"].mean()),
                "mean_recall": float(g["recall"].mean()),
                "mean_fpr": float(g["fpr"].mean()),
                "mean_roc_auc": float(g["roc_auc"].mean()),
                "mean_ms_per_window": float(g["ms_per_window"].mean()),
                "mean_detection_delay_ms": float(g["mean_delay_ms"].mean()),
                "mean_false_alarms_per_hour": float(g["false_alarms_per_hour"].mean()),
            })
    save_csv(out_root / "window_sensitivity_tradeoff.csv", pd.DataFrame(tradeoff_rows))
    return df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_phase1(cfg: dict, mode: str = "all") -> dict:
    set_global_seed(int(cfg.get("seed", 0)))
    out_root = Path(cfg.get("output_dir", "experiments/phase1"))
    out_root.mkdir(parents=True, exist_ok=True)
    save_json(out_root / "config.json", cfg)

    results: dict[str, str | int] = {"output_dir": str(out_root)}
    if mode in ("all", "cross_condition"):
        df = run_cross_condition(cfg, out_root / "cross_condition")
        results["cross_condition_rows"] = len(df)
    if mode in ("all", "global_threshold"):
        df = run_global_threshold(cfg, out_root / "global_threshold")
        results["global_threshold_rows"] = len(df)
    if mode in ("all", "residualization"):
        df = run_residualization(cfg, out_root / "residualization")
        results["residualization_rows"] = len(df)
    if mode in ("all", "window_sensitivity"):
        df = run_window_sensitivity(cfg, out_root / "window_sensitivity")
        results["window_sensitivity_rows"] = len(df)

    save_json(out_root / "run_summary.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CANguard Phase 1 — PIRD validation")
    parser.add_argument(
        "--config",
        default="experiments/configs/phase1_pird_validation.yaml",
        help="Path to phase 1 YAML config",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "cross_condition", "global_threshold",
                 "residualization", "window_sensitivity"],
        default="all",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_phase1(cfg, mode=args.mode)
    print("Phase 1 complete. Results:", result)


if __name__ == "__main__":
    main()
