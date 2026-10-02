"""Phase 2 runner: frozen PIRD baseline + white-box adaptive attacks.

Modes
-----
baseline : run the existing PIRD pipeline and freeze its configuration, metrics,
           scores, predictions and labels under ``<output_dir>/baseline``.
adaptive : pre-registered fixed-severity white-box adaptive attack sweep on
           legitimate IDs, plus the detection-boundary table.
gradual  : gradual / slow-drift adaptive attack with detection-latency analysis.
all      : baseline + adaptive + gradual.

Examples
--------
    python -m experiments.runners.run_phase2 \
        --config experiments/configs/phase2_baseline.yaml --mode baseline

    python -m experiments.runners.run_phase2 \
        --config experiments/configs/phase2_adaptive.yaml --mode adaptive
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.attacks import (  # noqa: E402
    AttackParameters,
    apply_alpha_series,
    apply_fixed_attack,
    gradual_alphas,
    resolve_target_features,
)
from canguard.detectors import create_detector  # noqa: E402
from canguard.evaluation import (  # noqa: E402
    compute_metrics,
    train_anomaly_detector,
)
from canguard.evaluation.boundary import (  # noqa: E402
    detection_probability,
    summarize_boundary,
)
from canguard.exp import (  # noqa: E402
    FeatureCache,
    load_config,
    save_csv,
    save_json,
    set_global_seed,
)
from canguard.exp.matrix import build_window_table, resolve_data_path  # noqa: E402
from canguard.features import (  # noqa: E402
    BEHAVIORAL_FEATURES_V1,
    fit_per_id_stats,
    temporal_split,
    transform_residuals,
)
from canguard.visualization.adaptive import (  # noqa: E402
    plot_detection_probability,
    plot_latency_vs_severity,
    plot_residual_distributions,
    plot_residual_trajectory,
)

NORMALIZATION = "per_id_zscore_calib_normal"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_ROOT,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _feature_cols(cfg: dict) -> list[str]:
    cols = cfg.get("feature_cols")
    return list(cols) if cols else list(BEHAVIORAL_FEATURES_V1)


def _make_detector(cfg: dict):
    det_cfg = dict(cfg.get("detector", {}))
    kind = det_cfg.pop("kind", "isolation_forest")
    return create_detector(kind, **det_cfg)


def _experiment_meta(cfg: dict, mode: str, seed: int) -> dict:
    return {
        "experiment_id": cfg.get("experiment_id", f"phase2_{mode}"),
        "phase": 2,
        "mode": mode,
        "dataset": cfg.get("dataset", "hcrl"),
        "datasets": list(cfg.get("datasets", [])),
        "split": cfg.get("split", {}),
        "seed": int(seed),
        "window_size": cfg.get("window_size", 30),
        "sample_size": cfg.get("sample_size"),
        "normalization": NORMALIZATION,
        "detector": cfg.get("detector", {}),
        "evaluation": cfg.get("evaluation", {}),
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# PIRD preparation (reuses existing pipeline primitives; PIRD unchanged)
# ---------------------------------------------------------------------------
def _prepare_dataset(name: str, cfg: dict, cache: FeatureCache) -> dict:
    feature_cols = _feature_cols(cfg)
    split = cfg["split"]
    eval_cfg = cfg.get("evaluation", {})

    csv_path = resolve_data_path(cfg.get("data_dir", "data"), name)
    ft = build_window_table(
        csv_path,
        window_size=int(cfg.get("window_size", 30)),
        sample_size=cfg.get("sample_size"),
        cache=cache,
    )
    calib, train, test = temporal_split(
        ft, split["calib_frac"], split["train_frac"], split["test_frac"]
    )
    # Per-ID statistics: calibration NORMALS ONLY (frozen).
    calib_norm = calib[calib["is_attack"] == 0]
    stats, gstats = fit_per_id_stats(calib, feature_cols)
    res_cols = [c + "_res" for c in feature_cols]

    res_train = transform_residuals(train, stats, gstats, feature_cols)
    res_test = transform_residuals(test, stats, gstats, feature_cols)

    detector = _make_detector(cfg)
    out = train_anomaly_detector(
        detector,
        res_train,
        res_test,
        res_cols,
        val_holdout_fraction=eval_cfg.get("val_holdout_fraction", 0.2),
        fpr_target=eval_cfg.get("fpr_target", 0.01),
    )
    return {
        "name": name,
        "feature_cols": feature_cols,
        "res_cols": res_cols,
        "calib": calib,
        "train": train,
        "test": test,
        "calib_norm": calib_norm,
        "stats": stats,
        "gstats": gstats,
        "detector": out["model"],
        "threshold": float(out["threshold"]),
        "baseline": out,
    }


def select_legit_targets(
    calib: pd.DataFrame,
    stats: dict,
    *,
    top_k: int = 3,
    min_calib_windows: int = 50,
    max_calib_attack_frac: float = 0.0,
    explicit: list[str] | None = None,
) -> list[str]:
    """Pick legitimate target IDs using **calibration labels only**.

    A target must (a) have per-ID statistics fitted, (b) have at least
    ``min_calib_windows`` normal windows in calibration, and (c) have an attack
    fraction in calibration no greater than ``max_calib_attack_frac``.
    """
    norm = calib[calib["is_attack"] == 0]
    atk = calib[calib["is_attack"] == 1]
    n_norm = norm["can_id"].astype(str).value_counts()
    n_atk = atk["can_id"].astype(str).value_counts()
    if explicit:
        out = []
        for cid in explicit:
            cid = str(cid)
            if cid in stats and int(n_norm.get(cid, 0)) >= min_calib_windows:
                out.append(cid)
        return out
    candidates: list[tuple[str, int]] = []
    for cid, n in n_norm.items():
        cid = str(cid)
        if cid not in stats or n < min_calib_windows:
            continue
        total = int(n) + int(n_atk.get(cid, 0))
        af = (int(n_atk.get(cid, 0)) / total) if total else 0.0
        if af > max_calib_attack_frac:
            continue
        candidates.append((cid, int(n)))
    candidates.sort(key=lambda t: (-t[1], t[0]))
    return [c for c, _ in candidates[:top_k]]


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def run_baseline(cfg: dict, out_root: Path) -> pd.DataFrame:
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase2"))
    out_dir = out_root / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", cfg)

    rows = []
    for name in cfg["datasets"]:
        prep = _prepare_dataset(name, cfg, cache)
        out = prep["baseline"]
        test = prep["test"]
        n_norm = int((test["is_attack"] == 0).sum())
        n_atk = int((test["is_attack"] == 1).sum())
        row = {
            "experiment_id": cfg.get("experiment_id", "phase2_baseline"),
            "dataset": name,
            "split": "40/20/40",
            "seed": seed,
            "window_size": cfg.get("window_size", 30),
            "normalization": NORMALIZATION,
            "detector": cfg.get("detector", {}).get("kind", "isolation_forest"),
            "threshold": out["threshold"],
            "n_windows": len(prep["test"]) + len(prep["train"]) + len(prep["calib"]),
            "n_test": len(test),
            "n_test_normal": n_norm,
            "n_test_attack": n_atk,
            "n_train_normals": out["n_train_normals"],
            "precision": out["precision"],
            "recall": out["recall"],
            "f1": out["f1"],
            "fpr": out["fpr"],
            "roc_auc": out["roc_auc"],
            "pr_auc": out["pr_auc"],
            "tp": out["tp"],
            "fp": out["fp"],
            "fn": out["fn"],
            "tn": out["tn"],
            "feature_cols": ",".join(prep["feature_cols"]),
        }
        rows.append(row)
        # Per-window predictions/scores (test segment).
        pd.DataFrame(
            {
                "timestamp": test["timestamp"].values,
                "can_id": test["can_id"].astype(str).values,
                "y_true": out["y_test"],
                "y_pred": out["y_pred"],
                "score": out["scores_test"],
                "is_attack": test["is_attack"].values,
            }
        ).to_csv(out_dir / f"predictions_{name}.csv", index=False)
        print(
            f"[baseline] {name}: F1={row['f1']:.3f} Recall={row['recall']:.3f} "
            f"FPR={row['fpr']:.4f} PR-AUC={row['pr_auc']:.3f}"
        )

    df = pd.DataFrame(rows)
    save_csv(out_dir / "metrics.csv", df)
    save_json(out_dir / "metrics.json", df.to_dict(orient="records"))
    save_json(
        out_dir / "summary.json",
        {"meta": _experiment_meta(cfg, "baseline", seed), "metrics": df.to_dict(orient="records")},
    )
    return df


# ---------------------------------------------------------------------------
# Adaptive fixed-severity sweep
# ---------------------------------------------------------------------------
def _score_residuals(detector, res_df: pd.DataFrame, res_cols: list[str]) -> np.ndarray:
    return detector.score_samples(res_df[res_cols].fillna(0).values)


def _two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """One-sided two-proportion z-test (attack rate > control rate).

    Returns ``(z, p_value)``. With fewer than 2 observations on either side the
    test is undefined and ``(0.0, 1.0)`` is returned.
    """
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2))
    if se <= 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 0.5 * math.erfc(z / math.sqrt(2.0))
    return float(z), float(p_val)


def run_adaptive(cfg: dict, out_root: Path) -> dict:
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase2"))
    out_dir = out_root / "adaptive_attack"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", cfg)

    atk_cfg = cfg["attack"]
    severities = [float(a) for a in atk_cfg["severities"]]
    directions = list(atk_cfg.get("directions", ["positive", "negative"]))
    modes = list(atk_cfg.get("attack_modes", ["mean_shift"]))
    feature_spec = atk_cfg.get("target_features", "all_attackable")
    targeting = cfg.get("targeting", {})

    sweep_rows: list[dict] = []
    normal_norm_rows: list[dict] = []
    residual_dist_rows: list[dict] = []
    targets_by_dataset: dict[str, list[str]] = {}
    boundary_example_max = float(atk_cfg.get("example_severity", 1.0))

    for name in cfg["datasets"]:
        prep = _prepare_dataset(name, cfg, cache)
        feature_cols = prep["feature_cols"]
        res_cols = prep["res_cols"]
        stats, gstats = prep["stats"], prep["gstats"]
        detector, threshold = prep["detector"], prep["threshold"]
        target_features = resolve_target_features(feature_spec, feature_cols)

        params = AttackParameters.from_stats(
            stats,
            gstats,
            feature_cols,
            provenance="calib_normal",
            n_calib_normal_windows=len(prep["calib_norm"]),
        )
        targets = select_legit_targets(
            prep["calib"],
            stats,
            top_k=int(targeting.get("top_k", 3)),
            min_calib_windows=int(targeting.get("min_calib_windows", 50)),
            max_calib_attack_frac=float(targeting.get("max_calib_attack_frac", 0.0)),
            explicit=targeting.get("target_ids"),
        )
        targets_by_dataset[name] = targets
        print(f"[adaptive] {name}: legitimate target IDs = {targets}")

        test_norm = prep["test"][prep["test"]["is_attack"] == 0].copy()
        test_norm = test_norm.sort_values("timestamp").reset_index(drop=True)
        res_norm = transform_residuals(test_norm, stats, gstats, feature_cols)
        scores_norm = _score_residuals(detector, res_norm, res_cols)
        flags_norm = scores_norm >= threshold
        ids = test_norm["can_id"].astype(str).values
        norm_norms = np.linalg.norm(res_norm[res_cols].to_numpy(dtype=float), axis=1)

        # normal residual-norm sample (for distribution figure)
        idx_n = np.linspace(0, len(norm_norms) - 1, min(5000, len(norm_norms))).astype(int)
        for v in norm_norms[idx_n]:
            normal_norm_rows.append({"dataset": name, "kind": "normal", "residual_norm": float(v)})

        for target in targets:
            positions = np.flatnonzero(ids == target)
            if len(positions) == 0:
                continue
            comp = np.setdiff1d(np.arange(len(ids)), positions)
            # Matched control: the same target windows, unmodified.
            control_flags = flags_norm[positions]
            control_n = int(len(positions))
            control_k = int(control_flags.sum())
            control_rate = control_k / control_n if control_n else float("nan")
            for mode in modes:
                for direction in directions:
                    for alpha in severities:
                        att_raw = test_norm.iloc[positions].copy()
                        att_raw = apply_fixed_attack(
                            params, att_raw, target, alpha, direction, target_features, mode=mode
                        )
                        res_att = transform_residuals(att_raw, stats, gstats, feature_cols)
                        scores_att = _score_residuals(detector, res_att, res_cols)
                        flags_att = scores_att >= threshold
                        att_norms = np.linalg.norm(
                            res_att[res_cols].to_numpy(dtype=float), axis=1
                        )

                        y_true = np.concatenate(
                            [np.zeros(len(comp), dtype=int), np.ones(len(positions), dtype=int)]
                        )
                        y_score = np.concatenate([scores_norm[comp], scores_att])
                        y_pred = (y_score >= threshold).astype(int)
                        m = compute_metrics(y_true, y_pred, y_score)
                        dp = detection_probability(flags_att)
                        z_stat, p_val = _two_proportion_z(
                            dp["n_detected"], len(positions), control_k, control_n
                        )
                        detected_significant = bool(
                            dp["detection_probability"] > control_rate and p_val < 0.05
                        )

                        sweep_rows.append(
                            {
                                "experiment_id": cfg.get("experiment_id", "phase2_adaptive"),
                                "dataset": name,
                                "target_id": target,
                                "attack_type": "adaptive_legitimate_id",
                                "attack_mode": mode,
                                "direction": direction,
                                "severity": alpha,
                                "feature_set": (
                                    feature_spec if isinstance(feature_spec, str) else "explicit"
                                ),
                                "target_features": ",".join(target_features),
                                "n_attack": int(len(positions)),
                                "n_normal": int(len(comp)),
                                "n_detected": dp["n_detected"],
                                "detection_probability": dp["detection_probability"],
                                "detection_prob_ci_low": dp["detection_prob_ci_low"],
                                "detection_prob_ci_high": dp["detection_prob_ci_high"],
                                "detected_significant": detected_significant,
                                "target_control_fpr": control_rate,
                                "target_control_n_flagged": control_k,
                                "two_prop_z": z_stat,
                                "p_value_onesided": p_val,
                                "precision": m["precision"],
                                "recall": m["recall"],
                                "f1": m["f1"],
                                "fpr": m["fpr"],
                                "roc_auc": m["roc_auc"],
                                "pr_auc": m["pr_auc"],
                                "threshold": threshold,
                                "max_residual_norm": float(np.max(att_norms)),
                                "mean_residual_norm": float(np.mean(att_norms)),
                            }
                        )

                        if abs(alpha - boundary_example_max) < 1e-12:
                            idx_a = np.linspace(
                                0, len(att_norms) - 1, min(3000, len(att_norms))
                            ).astype(int)
                            for v in att_norms[idx_a]:
                                residual_dist_rows.append(
                                    {
                                        "dataset": name,
                                        "kind": f"adaptive_{mode}_alpha_{alpha:g}_{direction}",
                                        "residual_norm": float(v),
                                    }
                                )

    sweep_df = pd.DataFrame(sweep_rows)
    save_csv(out_dir / "severity_sweep.csv", sweep_df)

    # Pooled boundary table by severity (all datasets/targets/directions/modes).
    boundary_df = _boundary_table(sweep_df)
    save_csv(out_dir / "boundary_table.csv", boundary_df)

    # Per attack-mode boundary tables.
    mode_rows = []
    for mode, g in sweep_df.groupby("attack_mode"):
        b = _boundary_table(g)
        b["attack_mode"] = mode
        mode_rows.append(b)
    if mode_rows:
        save_csv(out_dir / "boundary_table_by_mode.csv", pd.concat(mode_rows, ignore_index=True))

    # Per (attack mode, direction) boundary.
    dir_rows = []
    for (mode, direction), g in sweep_df.groupby(["attack_mode", "direction"]):
        b = _boundary_table(g)
        b["attack_mode"] = mode
        b["direction"] = direction
        dir_rows.append(b)
    if dir_rows:
        save_csv(
            out_dir / "boundary_table_by_direction.csv",
            pd.concat(dir_rows, ignore_index=True),
        )

    # Boundary summary per mode (alpha_50 / alpha_90), naive and control-adjusted.
    def _boundary_pair(b: pd.DataFrame) -> dict:
        sev = b["severity"].to_numpy(dtype=float)
        out = {"naive": summarize_boundary(sev, b["detection_probability"].to_numpy(dtype=float))}
        if "detection_probability_significant" in b.columns:
            out["significant"] = summarize_boundary(
                sev, b["detection_probability_significant"].to_numpy(dtype=float)
            )
        return out

    boundary_summary: dict[str, dict] = {}
    for mode, g in sweep_df.groupby("attack_mode"):
        boundary_summary[mode] = _boundary_pair(_boundary_table(g))
    pooled = _boundary_pair(boundary_df)
    save_json(
        out_dir / "summary.json",
        {
            "meta": _experiment_meta(cfg, "adaptive", seed),
            "attack": atk_cfg,
            "targets": targets_by_dataset,
            "note": (
                "'naive' detection is the raw flagged fraction and is inflated by "
                "each target ID's pre-existing control alarm rate. 'significant' "
                "detection requires the attack alarm rate to exceed the matched "
                "unmodified control (one-sided two-proportion z-test, p<0.05)."
            ),
            "boundary_pooled": pooled,
            "boundary_by_mode": boundary_summary,
            "boundary_table": boundary_df.to_dict(orient="records"),
        },
    )
    # Residual distribution data (normal vs adaptive).
    pd.DataFrame(normal_norm_rows + residual_dist_rows).to_csv(
        out_dir / "residual_norm_distributions.csv", index=False
    )
    return {
        "sweep": sweep_df,
        "boundary": boundary_df,
        "summary": {"pooled": pooled, "by_mode": boundary_summary},
    }


def _boundary_table(sweep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    has_sig = "detected_significant" in sweep_df.columns
    has_ctrl = "target_control_fpr" in sweep_df.columns
    for sev, g in sweep_df.groupby("severity"):
        n_att = int(g["n_attack"].sum())
        n_det = int(g["n_detected"].sum())
        dp = detection_probability(
            np.concatenate([np.ones(n_det, int), np.zeros(n_att - n_det, int)])
        )
        # Window-weighted metrics (aggregate confusion counts, then derive).
        tp = n_det
        fn = n_att - n_det
        fp = int((g["fpr"] * g["n_normal"]).round().sum())
        tn = int(g["n_normal"].sum()) - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        row = {
            "severity": float(sev),
            "n_attack": n_att,
            "n_detected": n_det,
            "detection_probability": dp["detection_probability"],
            "detection_prob_ci_low": dp["detection_prob_ci_low"],
            "detection_prob_ci_high": dp["detection_prob_ci_high"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "pr_auc": float(g["pr_auc"].mean(skipna=True)),
            "roc_auc": float(g["roc_auc"].mean(skipna=True)),
            "mean_max_residual_norm": float(g["max_residual_norm"].mean()),
        }
        if has_sig:
            row["detection_probability_significant"] = float(g["detected_significant"].mean())
        if has_ctrl:
            row["mean_target_control_fpr"] = float(g["target_control_fpr"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("severity").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gradual / slow drift
# ---------------------------------------------------------------------------
def run_gradual(cfg: dict, out_root: Path) -> dict:
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase2"))
    out_dir = out_root / "gradual_drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", cfg)

    atk_cfg = cfg["attack"]
    drift_cfg = atk_cfg.get("drift", {})
    severities = [float(a) for a in atk_cfg["severities"]]
    directions = list(atk_cfg.get("directions", ["positive"]))
    modes = list(atk_cfg.get("attack_modes", ["mean_shift"]))
    rates = list(drift_cfg.get("rates", ["fast", "medium", "slow"]))
    shape = drift_cfg.get("shape", "linear")
    fractions = drift_cfg.get("fractions")
    horizon = int(drift_cfg.get("horizon_windows", 300))
    feature_spec = atk_cfg.get("target_features", "all_attackable")
    targeting = cfg.get("targeting", {})
    save_trajectories = bool(drift_cfg.get("save_trajectories", True))

    result_rows: list[dict] = []
    traj_rows: list[dict] = []

    for name in cfg["datasets"]:
        prep = _prepare_dataset(name, cfg, cache)
        feature_cols = prep["feature_cols"]
        res_cols = prep["res_cols"]
        stats, gstats = prep["stats"], prep["gstats"]
        detector, threshold = prep["detector"], prep["threshold"]
        target_features = resolve_target_features(feature_spec, feature_cols)
        params = AttackParameters.from_stats(
            stats,
            gstats,
            feature_cols,
            provenance="calib_normal",
            n_calib_normal_windows=len(prep["calib_norm"]),
        )
        targets = select_legit_targets(
            prep["calib"],
            stats,
            top_k=int(targeting.get("top_k", 3)),
            min_calib_windows=int(targeting.get("min_calib_windows", 50)),
            max_calib_attack_frac=float(targeting.get("max_calib_attack_frac", 0.0)),
            explicit=targeting.get("target_ids"),
        )
        test_norm = prep["test"][prep["test"]["is_attack"] == 0].copy()
        test_norm = test_norm.sort_values("timestamp").reset_index(drop=True)
        res_norm = transform_residuals(test_norm, stats, gstats, feature_cols)
        scores_norm = _score_residuals(detector, res_norm, res_cols)
        flags_all = scores_norm >= threshold
        ids = test_norm["can_id"].astype(str).values

        for target in targets:
            positions = np.flatnonzero(ids == target)
            if len(positions) <= 5:
                continue
            if len(positions) <= horizon:
                block = positions
            else:
                start = (len(positions) - horizon) // 2
                block = positions[start : start + horizon]
            comp = np.setdiff1d(np.arange(len(ids)), block)
            fpr = float(flags_all[comp].mean()) if len(comp) else float("nan")
            # Matched control: the *same* windows, unmodified. This is the
            # false-alarm baseline the attack must beat.
            control_flags = flags_all[block]
            control_n = int(len(block))
            control_k = int(control_flags.sum())
            control_rate = (control_k / control_n) if control_n else float("nan")

            for mode in modes:
                for direction in directions:
                    for alpha_end in severities:
                        for rate in rates:
                            alphas = gradual_alphas(
                                len(block),
                                alpha_start=0.0,
                                alpha_end=alpha_end,
                                shape=shape,
                                rate=rate,
                                fractions=fractions,
                            )
                            att_raw = test_norm.iloc[block].copy()
                            att_raw = apply_alpha_series(
                                params,
                                att_raw,
                                target,
                                alphas,
                                direction,
                                target_features,
                                mode=mode,
                            )
                            res_att = transform_residuals(att_raw, stats, gstats, feature_cols)
                            scores_att = _score_residuals(detector, res_att, res_cols)
                            norms = np.linalg.norm(
                                res_att[res_cols].to_numpy(dtype=float), axis=1
                            )
                            flags = scores_att >= threshold
                            n_flagged = int(flags.sum())
                            if n_flagged:
                                first = int(np.argmax(flags))
                                delay = first  # windows after attack start (index 0)
                                delay_ms = float(
                                    att_raw["timestamp"].iloc[first]
                                    - att_raw["timestamp"].iloc[0]
                                ) * 1000.0
                                alpha_at_detect = float(alphas[first])
                            else:
                                first = -1
                                delay = float("nan")
                                delay_ms = float("nan")
                                alpha_at_detect = float("nan")

                            attack_n = int(len(flags))
                            attack_k = n_flagged
                            attack_rate = (attack_k / attack_n) if attack_n else float("nan")
                            z_stat, p_val = _two_proportion_z(
                                attack_k, attack_n, control_k, control_n
                            )
                            detected_significant = bool(attack_rate > control_rate and p_val < 0.05)

                            result_rows.append(
                                {
                                    "experiment_id": cfg.get("experiment_id", "phase2_gradual"),
                                    "dataset": name,
                                    "target_id": target,
                                    "attack_type": "gradual_adaptive",
                                    "attack_mode": mode,
                                    "direction": direction,
                                    "rate": rate,
                                    "shape": shape,
                                    "alpha_end": alpha_end,
                                    "horizon": len(block),
                                    "detected": bool(n_flagged),
                                    "detected_significant": detected_significant,
                                    "n_flagged": n_flagged,
                                    "delay_windows": delay,
                                    "delay_ms": delay_ms,
                                    "alpha_at_first_detect": alpha_at_detect,
                                    "attack_alarm_rate": attack_rate,
                                    "control_alarm_rate": control_rate,
                                    "control_n": control_n,
                                    "control_n_flagged": control_k,
                                    "alarm_lift": (
                                        attack_rate - control_rate
                                        if np.isfinite(attack_rate) and np.isfinite(control_rate)
                                        else float("nan")
                                    ),
                                    "two_prop_z": z_stat,
                                    "p_value_onesided": p_val,
                                    "max_residual_norm": float(np.max(norms)),
                                    "mean_residual_norm": float(np.mean(norms)),
                                    "cumulative_excess_score": float(
                                        np.sum(np.clip(scores_att - threshold, 0.0, None))
                                    ),
                                    "fpr_on_untouched": fpr,
                                    "threshold": threshold,
                                    "target_features": ",".join(target_features),
                                }
                            )
                            if save_trajectories:
                                for t in range(len(block)):
                                    traj_rows.append(
                                        {
                                            "dataset": name,
                                            "target_id": target,
                                            "attack_mode": mode,
                                            "direction": direction,
                                            "rate": rate,
                                            "shape": shape,
                                            "alpha_end": alpha_end,
                                            "t": t,
                                            "alpha": float(alphas[t]),
                                            "score": float(scores_att[t]),
                                            "residual_norm": float(norms[t]),
                                            "flagged": bool(flags[t]),
                                            "attack_start": 0,
                                            "threshold": threshold,
                                        }
                                    )

    results_df = pd.DataFrame(result_rows)
    save_csv(out_dir / "drift_results.csv", results_df)
    if save_trajectories and traj_rows:
        pd.DataFrame(traj_rows).to_csv(out_dir / "trajectories.csv", index=False)

    latency = _latency_table(results_df)
    save_csv(out_dir / "latency_table.csv", latency)

    drift_boundary: dict[str, dict] = {}
    by_sev_rows = []
    if len(results_df):
        if "attack_mode" in results_df.columns:
            mode_groups = results_df.groupby("attack_mode")
        else:
            mode_groups = [("all", results_df)]
        for mode, gm in mode_groups:
            agg = (
                gm.groupby("alpha_end")
                .agg(n=("detected", "size"), naive=("detected", "sum"),
                     significant=("detected_significant", "sum"))
                .reset_index()
            )
            for _, r in agg.iterrows():
                by_sev_rows.append(
                    {
                        "attack_mode": mode,
                        "alpha_end": float(r["alpha_end"]),
                        "n": int(r["n"]),
                        "detection_probability_naive": float(r["naive"] / r["n"]),
                        "detection_probability_significant": float(r["significant"] / r["n"]),
                    }
                )
            drift_boundary[mode] = {
                "naive": summarize_boundary(
                    agg["alpha_end"].to_numpy(dtype=float),
                    (agg["naive"] / agg["n"]).to_numpy(dtype=float),
                ),
                "significant": summarize_boundary(
                    agg["alpha_end"].to_numpy(dtype=float),
                    (agg["significant"] / agg["n"]).to_numpy(dtype=float),
                ),
            }
    if by_sev_rows:
        save_csv(out_dir / "drift_boundary.csv", pd.DataFrame(by_sev_rows))

    save_json(
        out_dir / "summary.json",
        {
            "meta": _experiment_meta(cfg, "gradual", seed),
            "attack": atk_cfg,
            "note": (
                "'naive' detection = any alarm in the horizon; it is inflated by "
                "the per-window false-alarm rate (~FPR). 'significant' detection "
                "requires the attack-window alarm rate to exceed the matched-control "
                "alarm rate (one-sided two-proportion z-test, p<0.05)."
            ),
            "drift_boundary": drift_boundary,
        },
    )
    return {
        "results": results_df,
        "latency": latency,
        "trajectories": traj_rows,
        "boundary": drift_boundary,
    }


def _latency_table(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    rows = []
    group_cols = ["alpha_end", "rate"]
    if "attack_mode" in df.columns:
        group_cols = ["attack_mode"] + group_cols
    for key, g in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(group_cols, key))
        det = g[g["detected"]]
        rec.update(
            {
                "n": int(len(g)),
                "detection_probability": float(g["detected"].mean()),
                "detection_probability_significant": (
                    float(g["detected_significant"].mean())
                    if "detected_significant" in g.columns
                    else float("nan")
                ),
                "mean_attack_alarm_rate": float(g["attack_alarm_rate"].mean())
                if "attack_alarm_rate" in g.columns
                else float("nan"),
                "mean_control_alarm_rate": float(g["control_alarm_rate"].mean())
                if "control_alarm_rate" in g.columns
                else float("nan"),
                "median_delay_windows": (
                    float(det["delay_windows"].median()) if len(det) else float("nan")
                ),
                "p95_delay_windows": (
                    float(det["delay_windows"].quantile(0.95))
                    if len(det) >= 2
                    else float("nan")
                ),
                "median_delay_ms": float(det["delay_ms"].median()) if len(det) else float("nan"),
                "mean_max_residual_norm": float(g["max_residual_norm"].mean()),
                "mean_cumulative_excess_score": float(g["cumulative_excess_score"].mean()),
            }
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    sort_cols = (["attack_mode"] if "attack_mode" in out.columns else []) + ["rate", "alpha_end"]
    return out.sort_values(sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots + consolidated results
# ---------------------------------------------------------------------------
def _primary_mode(cfg: dict, df: pd.DataFrame | None) -> str | None:
    modes = cfg.get("attack", {}).get("attack_modes")
    if modes:
        return str(modes[0])
    if df is not None and "attack_mode" in df.columns and len(df):
        return str(df["attack_mode"].iloc[0])
    return None


def make_plots(cfg: dict, out_root: Path, adaptive: dict | None, gradual: dict | None) -> None:
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if adaptive is not None and len(adaptive.get("sweep", [])) > 0:
        sweep = adaptive["sweep"]
        primary = _primary_mode(cfg, sweep)
        # Pooled detection-probability curve.
        plot_detection_probability(
            adaptive["boundary"], plot_dir / "detection_probability_vs_severity.png"
        )
        # By attack mode (the two white-box parameterizations).
        mode_tables = []
        for mode, g in sweep.groupby("attack_mode"):
            b = _boundary_table(g)
            b["attack_mode"] = mode
            mode_tables.append(b)
        if mode_tables:
            mode_df = pd.concat(mode_tables, ignore_index=True)
            plot_detection_probability(
                mode_df,
                plot_dir / "detection_probability_by_mode.png",
                group_col="attack_mode",
            )
            if "detection_probability_significant" in mode_df.columns:
                plot_detection_probability(
                    mode_df,
                    plot_dir / "detection_probability_significant.png",
                    prob_col="detection_probability_significant",
                    group_col="attack_mode",
                    title="Adaptive attack boundary (control-adjusted)",
                )
        by_dir = adaptive.get("by_direction")
        if by_dir is not None and len(by_dir):
            if primary is not None and "attack_mode" in by_dir.columns:
                by_dir = by_dir[by_dir["attack_mode"] == primary]
            if len(by_dir):
                plot_detection_probability(
                    by_dir,
                    plot_dir / "detection_probability_by_direction.png",
                    group_col="direction",
                )
        # Residual distributions, primary mode only.
        dist_path = out_root / "adaptive_attack" / "residual_norm_distributions.csv"
        if dist_path.exists():
            dist = pd.read_csv(dist_path)
            if primary is not None and "kind" in dist.columns:
                marker = f"_{primary}_"
                filtered = dist[
                    dist["kind"].str.contains(marker, regex=False)
                    | (dist["kind"] == "normal")
                ]
                dist = filtered if len(filtered) else dist
            for ds in dist["dataset"].unique():
                sub = dist[dist["dataset"] == ds]
                normal = sub[sub["kind"] == "normal"]["residual_norm"].to_numpy(dtype=float)
                atk = sub[sub["kind"] != "normal"]["residual_norm"].to_numpy(dtype=float)
                if len(normal) and len(atk):
                    plot_residual_distributions(
                        normal,
                        atk,
                        plot_dir / f"residual_norm_distributions_{ds}.png",
                        title=f"{ds}: normal vs adaptive residual norms",
                    )
    if gradual is not None and len(gradual.get("results", [])) > 0:
        res = gradual["results"]
        traj_df = pd.DataFrame(gradual.get("trajectories", []))
        primary = _primary_mode(cfg, res if "attack_mode" in res.columns else traj_df)
        res_plot = (
            res[res["attack_mode"] == primary]
            if (primary and "attack_mode" in res.columns)
            else res
        )
        if len(res_plot):
            plot_latency_vs_severity(
                res_plot,
                plot_dir / "detection_latency_vs_severity.png",
                severity_col="alpha_end",
                latency_col="delay_windows",
                group_col="rate",
            )
        # Control-adjusted gradual detection probability (the naive curve is
        # dominated by the per-window false-alarm rate over the horizon).
        drift_boundary_path = out_root / "gradual_drift" / "drift_boundary.csv"
        if drift_boundary_path.exists():
            db = pd.read_csv(drift_boundary_path)
            if primary and "attack_mode" in db.columns:
                db = db[db["attack_mode"] == primary]
            if len(db):
                plot_detection_probability(
                    db,
                    plot_dir / "gradual_detection_probability_significant.png",
                    severity_col="alpha_end",
                    prob_col="detection_probability_significant",
                    title="Gradual drift: control-adjusted detection probability",
                )
        if len(traj_df):
            if primary and "attack_mode" in traj_df.columns:
                traj_df = traj_df[traj_df["attack_mode"] == primary]
            # One representative slow-drift trajectory.
            rep = traj_df[
                (traj_df["rate"] == "slow")
                & (traj_df["alpha_end"] == traj_df["alpha_end"].max())
            ]
            if len(rep):
                ds0 = rep["dataset"].iloc[0]
                rep = rep[rep["dataset"] == ds0]
                tgt0 = rep["target_id"].iloc[0]
                rep = rep[rep["target_id"] == tgt0]
                dir0 = rep["direction"].iloc[0]
                rep = rep[rep["direction"] == dir0]
                plot_residual_trajectory(
                    rep,
                    plot_dir / "example_gradual_drift_trajectory.png",
                    title=f"Gradual drift ({ds0}, id {tgt0}, {dir0})",
                )


# ---------------------------------------------------------------------------
# Comparison table (normal / non-adaptive / adaptive / gradual)
# ---------------------------------------------------------------------------
def _comparison_table(
    out_root: Path, adaptive_result: dict | None, gradual_result: dict | None
) -> pd.DataFrame:
    rows: list[dict] = []
    # Allow building the comparison from previously-saved artifacts when a
    # single mode was run in isolation (e.g. --mode gradual after --mode adaptive).
    if adaptive_result is None:
        p = out_root / "adaptive_attack" / "severity_sweep.csv"
        if p.exists():
            adaptive_result = {"sweep": pd.read_csv(p)}
    if gradual_result is None:
        p = out_root / "gradual_drift" / "drift_results.csv"
        if p.exists():
            gradual_result = {"results": pd.read_csv(p)}
    base_path = out_root / "baseline" / "metrics.csv"
    if base_path.exists():
        for _, r in pd.read_csv(base_path).iterrows():
            rows.append(
                {
                    "family": "normal_traffic",
                    "dataset": r["dataset"],
                    "severity": 0.0,
                    "detection_probability": r["fpr"],
                    "recall": r["recall"],
                    "fpr": r["fpr"],
                    "f1": r["f1"],
                    "n": int(r["n_test_normal"]),
                    "notes": "FPR on untouched test normals",
                }
            )
            rows.append(
                {
                    "family": "non_adaptive_existing",
                    "dataset": r["dataset"],
                    "severity": float("nan"),
                    "detection_probability": r["recall"],
                    "recall": r["recall"],
                    "fpr": r["fpr"],
                    "f1": r["f1"],
                    "n": int(r["n_test_attack"]),
                    "notes": "real HCRL injection in test split",
                }
            )
    if adaptive_result is not None and len(adaptive_result.get("sweep", [])) > 0:
        s = adaptive_result["sweep"]
        for (ds, mode, sev), g in s.groupby(["dataset", "attack_mode", "severity"]):
            n = int(g["n_attack"].sum())
            k = int(g["n_detected"].sum())
            naive = (k / n) if n else float("nan")
            sig = (
                float(g["detected_significant"].mean())
                if "detected_significant" in g.columns
                else float("nan")
            )
            rows.append(
                {
                    "family": f"adaptive_fixed_{mode}",
                    "dataset": ds,
                    "severity": float(sev),
                    "detection_probability": sig,
                    "detection_probability_naive": naive,
                    "recall": sig,
                    "fpr": float((g["fpr"] * g["n_normal"]).sum() / g["n_normal"].sum()),
                    "f1": float(g["f1"].mean()),
                    "n": n,
                    "notes": "control-adjusted significant white-box fixed severity",
                }
            )
    if gradual_result is not None and len(gradual_result.get("results", [])) > 0:
        gr = gradual_result["results"]
        for (ds, mode, ae), g in gr.groupby(["dataset", "attack_mode", "alpha_end"]):
            rows.append(
                {
                    "family": f"gradual_{mode}",
                    "dataset": ds,
                    "severity": float(ae),
                    "detection_probability": float(g["detected_significant"].mean()),
                    "recall": float(g["detected_significant"].mean()),
                    "fpr": float(g["fpr_on_untouched"].mean()),
                    "f1": float("nan"),
                    "n": int(len(g)),
                    "notes": "control-adjusted significant gradual drift",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CANguard Phase 2 runner")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["baseline", "adaptive", "gradual", "hybrid", "shift", "road", "all"],
        default="all",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg.get("output_dir", "experiments/phase2"))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase 2 mode={args.mode} config={args.config} ===")

    adaptive_result = None
    gradual_result = None

    if args.mode == "shift":
        from experiments.runners.phase2_shift import run_global_shift  # lazy

        shift_root = out_root / "global_shift"
        run_global_shift(cfg, shift_root, plot_dir=out_root / "plots")
        results_dir = out_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for fname in [
            "feature_shift.csv",
            "family_shift.csv",
            "score_shift.csv",
            "flag_association.csv",
            "calibration_coverage.csv",
        ]:
            src = shift_root / fname
            if src.exists():
                save_csv(results_dir / f"shift_{fname}", pd.read_csv(src))
        print(f"\nPhase 2 global-shift diagnosis complete. Results under {shift_root}")
        return

    if args.mode == "road":
        from experiments.runners.phase2_road import run_road_external  # lazy

        out = Path(cfg.get("output_dir", "experiments/phase2/road_external"))
        run_road_external(cfg, out)
        results_dir = out.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for rel in [
            "road_attack_summary.csv",
            "hybrid/conventional_metrics.csv",
            "hybrid/fixed_fpr.csv",
            "hybrid/adaptive_boundary.csv",
            "hybrid/complementarity.csv",
            "hybrid/score_correlation.csv",
            "hybrid/ablation_conventional.csv",
        ]:
            src = out / rel
            if src.exists():
                fname = Path(rel).name
                dest = results_dir / (fname if fname.startswith("road_") else f"road_{fname}")
                save_csv(dest, pd.read_csv(src))
        print(f"\nPhase 2 ROAD external validation complete. Results under {out}")
        return

    if args.mode == "hybrid":
        if "global" not in cfg:
            raise SystemExit("Config lacks a 'global' section required for hybrid mode.")
        from canguard.visualization.hybrid import make_hybrid_plots
        from experiments.runners.phase2_hybrid import run_hybrid  # lazy to avoid cycle

        hybrid_result = run_hybrid(cfg, out_root)
        make_hybrid_plots(cfg, out_root, hybrid_result)
        results_dir = out_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        for rel in [
            "hybrid/conventional_metrics.csv",
            "hybrid/fixed_fpr.csv",
            "hybrid/fpr_recall_curve.csv",
            "hybrid/adaptive_boundary.csv",
            "hybrid/adaptive_sweep.csv",
            "hybrid/gradual_boundary.csv",
            "hybrid/complementarity.csv",
            "hybrid/score_correlation.csv",
            "hybrid/ablation.csv",
            "hybrid/ablation_conventional.csv",
            "global/conventional_metrics.csv",
            "global/adaptive_boundary.csv",
        ]:
            src = out_root / rel
            if src.exists():
                # Prefix with the source subdirectory to avoid collisions
                # (e.g. hybrid vs global conventional_metrics.csv).
                dest = results_dir / f"{Path(rel).parent.name}_{Path(rel).name}"
                save_csv(dest, pd.read_csv(src))
        print(f"\nPhase 2 hybrid complete. Results under {out_root}")
        return

    if args.mode in ("baseline", "all"):
        run_baseline(cfg, out_root)
    if args.mode in ("adaptive", "all"):
        if "attack" not in cfg:
            raise SystemExit("Config lacks an 'attack' section required for adaptive mode.")
        adaptive_result = run_adaptive(cfg, out_root)
        # stash per-direction table for plotting
        by_dir_path = out_root / "adaptive_attack" / "boundary_table_by_direction.csv"
        if by_dir_path.exists():
            adaptive_result["by_direction"] = pd.read_csv(by_dir_path)
    if args.mode in ("gradual", "all"):
        if "attack" not in cfg:
            raise SystemExit("Config lacks an 'attack' section required for gradual mode.")
        gradual_result = run_gradual(cfg, out_root)

    make_plots(cfg, out_root, adaptive_result, gradual_result)

    # Consolidated results directory.
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "baseline/metrics.csv",
        "adaptive_attack/severity_sweep.csv",
        "adaptive_attack/boundary_table.csv",
        "adaptive_attack/boundary_table_by_mode.csv",
        "adaptive_attack/boundary_table_by_direction.csv",
        "gradual_drift/drift_results.csv",
        "gradual_drift/latency_table.csv",
        "gradual_drift/drift_boundary.csv",
    ]:
        src = out_root / rel
        if src.exists():
            save_csv(results_dir / Path(rel).name, pd.read_csv(src))

    comparison = _comparison_table(out_root, adaptive_result, gradual_result)
    if len(comparison):
        save_csv(results_dir / "comparison_table.csv", comparison)

    summary = {"config": cfg, "git_commit": _git_commit()}
    if adaptive_result is not None:
        summary["adaptive_boundary"] = adaptive_result.get("summary", {})
    if gradual_result is not None:
        summary["gradual_boundary"] = gradual_result.get("boundary", {})
    save_json(results_dir / "phase2_summary.json", summary)
    print(f"\nPhase 2 complete. Results under {out_root}")


if __name__ == "__main__":
    main()
