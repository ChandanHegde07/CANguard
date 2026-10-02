"""Part B — diagnose the global representation's normal-distribution shift.

Uses ONLY normal traffic for the calibration-vs-test shift analysis. Compares
the frozen calibration-normal global feature distribution with the frozen
test-normal distribution, ranks features by effect size, aggregates by family,
relates shifted features to false alarms, and runs one clearly-labelled
diagnostic calibration-coverage experiment. The global model is NOT modified.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from canguard.evaluation.distribution_shift import (  # noqa: E402
    association_with_flags,
    family_shift_summary,
    feature_shift_table,
)
from canguard.exp import FeatureCache, save_csv, save_json, set_global_seed  # noqa: E402
from canguard.features import (  # noqa: E402
    ALL_GLOBAL_FEATURES,
    GLOBAL_FEATURE_FAMILIES,
)
from canguard.visualization.style import (  # noqa: E402
    IEEE_COLORS,
    apply_ieee_style,
    figsize_double,
    figsize_single,
    save_ieee_figure,
)

from .phase2_hybrid import _fit_global_branch, _prepare_hybrid_dataset  # noqa: E402
from .run_phase2 import _experiment_meta  # noqa: E402


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))
    return x, np.arange(1, len(x) + 1) / max(1, len(x))


def _plot_top_feature_ecdfs(
    cal: pd.DataFrame, test: pd.DataFrame, features: list[str], dataset: str, path: Path
) -> None:
    apply_ieee_style()
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize_double(1.9 * nrows), squeeze=False)
    for ax, f in zip(axes.flatten(), features):
        xc, yc = _ecdf(cal[f].to_numpy(dtype=float))
        xt, yt = _ecdf(test[f].to_numpy(dtype=float))
        ax.plot(xc, yc, color=IEEE_COLORS["normal"], label="calibration normal")
        ax.plot(xt, yt, color=IEEE_COLORS["attack"], label="test normal")
        ax.set_title(f, fontsize=7)
        ax.legend(fontsize=5)
    for ax in axes.flatten()[len(features):]:
        ax.set_visible(False)
    fig.suptitle(f"{dataset}: top shifted global features (ECDF)")
    fig.tight_layout()
    save_ieee_figure(fig, path)


def _plot_score_ecdf(
    val_scores: np.ndarray,
    test_normal_scores: np.ndarray,
    test_attack_scores: np.ndarray,
    threshold: float,
    dataset: str,
    path: Path,
) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=figsize_single(2.7))
    for arr, color, label in [
        (val_scores, IEEE_COLORS["normal"], "calibration (val) normal"),
        (test_normal_scores, IEEE_COLORS["raw"], "test normal"),
    ]:
        x, y = _ecdf(np.asarray(arr))
        ax.plot(x, y, color=color, label=label)
    if len(test_attack_scores):
        x, y = _ecdf(np.asarray(test_attack_scores))
        ax.plot(x, y, color=IEEE_COLORS["attack"], ls="--", label="test attack")
    ax.axvline(threshold, color=IEEE_COLORS["neutral"], ls=":", lw=1.0, label="threshold")
    ax.set_xlabel("Global anomaly score")
    ax.set_ylabel("ECDF")
    ax.set_title(f"{dataset}: global score shift")
    ax.legend(fontsize=6)
    return save_ieee_figure(fig, path)


def _plot_family_shift(fam: pd.DataFrame, dataset: str, path: Path) -> None:
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=figsize_single(2.5))
    ax.bar(fam["family"], fam["mean_abs_smd"], color=IEEE_COLORS["residual"])
    ax.set_ylabel("Mean |SMD|")
    ax.set_title(f"{dataset}: shift by feature family")
    ax.tick_params(axis="x", rotation=20)
    return save_ieee_figure(fig, path)


def _plot_flag_association(assoc: pd.DataFrame, dataset: str, path: Path) -> None:
    apply_ieee_style()
    top = assoc.dropna(subset=["abs_smd_flagged"]).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize_single(3.4))
    ax.barh(top["feature"], top["abs_smd_flagged"], color=IEEE_COLORS["attack"])
    ax.set_xlabel("|SMD| flagged vs non-flagged (normal)")
    ax.set_title(f"{dataset}: shift association with false alarms")
    return save_ieee_figure(fig, path)


def run_global_shift(cfg: dict, out_root: Path, plot_dir: Path | None = None) -> dict:
    """Run the normal-distribution shift diagnosis for each HCRL dataset."""
    set_global_seed(int(cfg.get("seed", 0)))
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase2"))
    out_root.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(plot_dir) if plot_dir is not None else out_root
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_root / "config.json", cfg)

    feature_rows, family_rows, score_rows, assoc_rows, coverage_rows = [], [], [], [], []
    gcfg = cfg.get("global", {})
    global_cols = list(gcfg.get("feature_cols") or ALL_GLOBAL_FEATURES)

    for name in cfg["datasets"]:
        prep = _prepare_hybrid_dataset(name, cfg, cache)
        cal_norm = prep["gcalib"][prep["gcalib"]["is_attack"] == 0].copy()
        test_all = prep["gtest"]
        test_norm = test_all[test_all["is_attack"] == 0].copy()
        y = test_all["is_attack"].to_numpy(dtype=int)
        scores = prep["global"]["metrics"]["scores_test"]
        threshold = prep["global"]["threshold"]

        # --- feature-level normal shift ---
        fs = feature_shift_table(cal_norm, test_norm, global_cols, GLOBAL_FEATURE_FAMILIES)
        fs.insert(0, "dataset", name)
        feature_rows.append(fs)
        fam = family_shift_summary(fs)
        fam.insert(0, "dataset", name)
        family_rows.append(fam)

        # --- score distribution shift (normal vs normal) ---
        val = np.asarray(prep["global"]["val"], dtype=float)
        test_normal_scores = scores[y == 0]
        test_attack_scores = scores[y == 1]
        actual_fpr = (
            float((test_normal_scores >= threshold).mean())
            if len(test_normal_scores)
            else float("nan")
        )
        score_rows.append({
            "dataset": name,
            "target_fpr": prep["fpr_target"],
            "actual_test_normal_fpr": actual_fpr,
            "threshold": threshold,
            "val_normal_mean": float(np.mean(val)),
            "val_normal_median": float(np.median(val)),
            "val_normal_q95": float(np.quantile(val, 0.95)) if len(val) else float("nan"),
            "test_normal_mean": (
                float(np.mean(test_normal_scores)) if len(test_normal_scores) else float("nan")
            ),
            "test_normal_median": (
                float(np.median(test_normal_scores)) if len(test_normal_scores) else float("nan")
            ),
            "test_attack_mean": (
                float(np.mean(test_attack_scores)) if len(test_attack_scores) else float("nan")
            ),
            "n_val_normal": int(len(val)),
            "n_test_normal": int(len(test_normal_scores)),
            "n_test_attack": int(len(test_attack_scores)),
        })

        # --- association of feature deviation with false alarms (normal only) ---
        flags = scores >= threshold
        mask = y == 0
        assoc = association_with_flags(
            cal_norm, test_norm, flags[mask], scores[mask], global_cols,
            GLOBAL_FEATURE_FAMILIES,
        )
        assoc.insert(0, "dataset", name)
        assoc_rows.append(assoc)

        # --- diagnostic: broader calibration coverage (calib + train normals) ---
        gcal = prep["gcalib"]
        gtr = prep["gtrain"]
        broad = pd.concat(
            [gcal[gcal["is_attack"] == 0], gtr[gtr["is_attack"] == 0]], ignore_index=True
        )
        method = gcfg.get("normalization", "robust")
        branch = _fit_global_branch(
            broad, broad, prep["gtest"], global_cols, cfg, method,
            prep["val_fraction"], prep["fpr_target"],
        )
        sc_b = branch["metrics"]["scores_test"]
        thr_b = branch["threshold"]
        coverage_rows.append({
            "dataset": name,
            "protocol": "primary_calib_only",
            "target_fpr": prep["fpr_target"],
            "actual_test_normal_fpr": actual_fpr,
            "recall": (
                float((scores[y == 1] >= threshold).mean())
                if (y == 1).any()
                else float("nan")
            ),
            "n_calibration_normals": int((prep["gcalib"]["is_attack"] == 0).sum()),
            "n_train_normals": int((prep["gtrain"]["is_attack"] == 0).sum()),
        })
        coverage_rows.append({
            "dataset": name,
            "protocol": "diagnostic_broad_calibration",
            "target_fpr": prep["fpr_target"],
            "actual_test_normal_fpr": (
                float((sc_b[y == 0] >= thr_b).mean()) if (y == 0).any() else float("nan")
            ),
            "recall": (
                float((sc_b[y == 1] >= thr_b).mean()) if (y == 1).any() else float("nan")
            ),
            "n_calibration_normals": int(len(broad)),
            "n_train_normals": int(len(broad)),
        })

        _plot_top_feature_ecdfs(
            cal_norm, test_norm, list(fs["feature"].head(6)), name,
            plot_dir / f"shift_top_features_ecdf_{name}.png",
        )
        _plot_score_ecdf(val, test_normal_scores, test_attack_scores, threshold, name,
                         plot_dir / f"shift_score_ecdf_{name}.png")
        _plot_family_shift(fam, name, plot_dir / f"shift_family_{name}.png")
        _plot_flag_association(assoc, name, plot_dir / f"shift_flag_association_{name}.png")

    feature_shift = pd.concat(feature_rows, ignore_index=True)
    family_shift = pd.concat(family_rows, ignore_index=True)
    score_shift = pd.DataFrame(score_rows)
    associations = pd.concat(assoc_rows, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)

    save_csv(out_root / "feature_shift.csv", feature_shift)
    save_csv(out_root / "family_shift.csv", family_shift)
    save_csv(out_root / "score_shift.csv", score_shift)
    save_csv(out_root / "flag_association.csv", associations)
    save_csv(out_root / "calibration_coverage.csv", coverage)
    save_json(
        out_root / "summary.json",
        {
            "meta": _experiment_meta(cfg, "shift", int(cfg.get("seed", 0))),
            "note": (
                "Feature shift uses ONLY calibration-normal vs test-normal traffic. "
                "'diagnostic_broad_calibration' is a labelled calibration-coverage "
                "diagnostic, not the primary frozen result."
            ),
            "top_shifted_features_per_dataset": {
                ds: g.head(5)[["feature", "family", "smd", "ks", "normalized_wasserstein"]]
                .to_dict(orient="records")
                for ds, g in feature_shift.groupby("dataset")
            },
        },
    )
    return {
        "feature_shift": feature_shift,
        "family_shift": family_shift,
        "score_shift": score_shift,
        "associations": associations,
        "coverage": coverage,
    }


__all__ = ["run_global_shift"]
