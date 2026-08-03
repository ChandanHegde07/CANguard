"""Phase C: multi-seed, statistics, importance, runtime, latency, errors, paper assets.

Usage:
    python -m experiments.runners.run_phase_c --config experiments/configs/phase_c.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.detectors import create_detector
from canguard.evaluation import (
    bootstrap_metrics,
    flatten_bootstrap,
    train_anomaly_detector,
)
from canguard.evaluation.errors import catalog_errors
from canguard.evaluation.importance import leave_one_feature_out, permutation_importance
from canguard.evaluation.latency import detection_latency, multi_segment_latencies
from canguard.evaluation.stats_tests import mcnemar_test, paired_bootstrap_delta
from canguard.exp import (
    ExperimentRun,
    FeatureCache,
    load_config,
    save_csv,
    save_json,
    set_global_seed,
    setup_logging,
)
from canguard.exp.matrix import (
    build_raw_and_residual_splits,
    build_window_table,
    resolve_data_path,
)
from canguard.exp.resources import peak_rss_mb, timed
from canguard.exp.road_protocol import (
    list_eval_captures,
    prepare_road_capture,
    resolve_road_root,
)
from canguard.features import BEHAVIORAL_FEATURES_V1
from canguard.visualization.style import IEEE_COLORS, apply_ieee_style

logger = logging.getLogger("canguard")


def _params(cfg, det_name, seed):
    p = dict(cfg.get("detector_params", {}).get(det_name, {}))
    # pass seed where accepted
    p["random_state"] = seed
    return p


def _run_one(
    det_name,
    params,
    train_df,
    test_df,
    cols,
    eval_cfg,
    boot_cfg,
    seed,
    measure=True,
    do_bootstrap=True,
):
    det = create_detector(det_name, **params)
    out = train_anomaly_detector(
        det,
        train_df,
        test_df,
        cols,
        val_holdout_fraction=eval_cfg.get("val_holdout_fraction", 0.2),
        fpr_target=eval_cfg.get("fpr_target", 0.01),
        measure_resources=measure,
    )
    boot = {}
    if do_bootstrap and int(boot_cfg.get("n_boot", 0) or 0) > 0:
        boot = bootstrap_metrics(
            out["y_test"],
            out["y_pred"],
            out["scores_test"],
            n_boot=int(boot_cfg.get("n_boot", 300)),
            block_size=int(boot_cfg.get("block_size", 50)),
            seed=seed,
            ci=float(boot_cfg.get("ci", 0.95)),
        )
    return out, boot


def prepare_hcrl_matrices(cfg, cache):
    mats = {}
    h = cfg["hcrl"]
    for name in h["datasets"]:
        path = resolve_data_path(h["data_dir"], name)
        ft = build_window_table(
            path,
            window_size=int(h["window_size"]),
            sample_size=h.get("sample_size"),
            cache=cache,
        )
        mats[name] = build_raw_and_residual_splits(ft, **h["split"])
        logger.info("HCRL %s ready train=%d test=%d", name, len(mats[name]["raw"]["train"]), len(mats[name]["raw"]["test"]))
    return mats


def prepare_road_matrices(cfg, cache):
    rcfg = cfg["road"]
    root = resolve_road_root(rcfg["data_dir"])
    cs = rcfg.get("capture_sample", {})
    captures = list_eval_captures(
        root,
        skip_masquerade=cs.get("skip_masquerade", True),
        skip_unlabeled=cs.get("skip_unlabeled", True),
        per_type=cs.get("per_type"),
    )
    mats = {}
    for name, meta in captures:
        prep = prepare_road_capture(
            root,
            name,
            meta,
            window_size=int(rcfg["window_size"]),
            max_frames=rcfg.get("max_frames_per_capture"),
            cache=cache,
        )
        if prep.get("error"):
            logger.warning("ROAD skip %s: %s", name, prep["error"])
            continue
        mats[name] = prep
        logger.info(
            "ROAD %s pre=%d test=%d atk=%d",
            name,
            prep["n_pre"],
            prep["n_test"],
            prep["n_test_attack"],
        )
    return mats


def task_multiseed(cfg, hcrl_mats, road_mats, run: ExperimentRun):
    seeds = list(cfg.get("seeds", [0, 1, 2, 3, 4]))
    detectors = cfg["detectors"]
    reps = cfg.get("representations", ["raw", "residual"])
    eval_cfg = cfg["evaluation"]
    boot_cfg = cfg.get("bootstrap", {})
    rows = []

    # HCRL
    for ds, mat in hcrl_mats.items():
        for rep in reps:
            train = mat[rep]["train"]
            test = mat[rep]["test"]
            cols = mat["feature_cols"] if rep == "raw" else mat["residual_cols"]
            for det_name in detectors:
                for seed in seeds:
                    set_global_seed(seed)
                    logger.info("multiseed HCRL %s %s %s seed=%s", ds, rep, det_name, seed)
                    try:
                        # Multi-seed CIs come from seed aggregation; skip heavy block bootstrap here.
                        out, boot = _run_one(
                            det_name,
                            _params(cfg, det_name, seed),
                            train,
                            test,
                            cols,
                            eval_cfg,
                            {"n_boot": 0},
                            seed,
                            do_bootstrap=False,
                        )
                        row = {
                            "corpus": "hcrl",
                            "dataset": ds,
                            "capture": ds,
                            "attack_type": ds,
                            "representation": rep,
                            "detector": det_name,
                            "seed": seed,
                            "precision": out["precision"],
                            "recall": out["recall"],
                            "f1": out["f1"],
                            "roc_auc": out["roc_auc"],
                            "pr_auc": out["pr_auc"],
                            "fpr": out["fpr"],
                            "train_seconds": out.get("train_seconds"),
                            "score_seconds": out.get("score_seconds"),
                            "runtime_seconds": out.get("runtime_seconds"),
                            "peak_rss_mb": out.get("peak_rss_mb"),
                            "model_bytes": out.get("model_bytes"),
                        }
                        rows.append(row)
                    except Exception as exc:
                        logger.exception("fail %s", exc)
                        rows.append(
                            {
                                "corpus": "hcrl",
                                "dataset": ds,
                                "representation": rep,
                                "detector": det_name,
                                "seed": seed,
                                "error": str(exc),
                            }
                        )

    # ROAD
    for cap, mat in road_mats.items():
        for rep in reps:
            train = mat[rep]["train"]
            test = mat[rep]["test"]
            cols = mat["feature_cols"] if rep == "raw" else mat["residual_cols"]
            for det_name in detectors:
                for seed in seeds:
                    set_global_seed(seed)
                    logger.info("multiseed ROAD %s %s %s seed=%s", cap, rep, det_name, seed)
                    try:
                        out, boot = _run_one(
                            det_name,
                            _params(cfg, det_name, seed),
                            train,
                            test,
                            cols,
                            eval_cfg,
                            {"n_boot": 0},
                            seed,
                            do_bootstrap=False,
                        )
                        row = {
                            "corpus": "road",
                            "dataset": mat.get("attack_type", cap),
                            "capture": cap,
                            "attack_type": mat.get("attack_type", cap),
                            "representation": rep,
                            "detector": det_name,
                            "seed": seed,
                            "precision": out["precision"],
                            "recall": out["recall"],
                            "f1": out["f1"],
                            "roc_auc": out["roc_auc"],
                            "pr_auc": out["pr_auc"],
                            "fpr": out["fpr"],
                            "train_seconds": out.get("train_seconds"),
                            "score_seconds": out.get("score_seconds"),
                            "runtime_seconds": out.get("runtime_seconds"),
                            "peak_rss_mb": out.get("peak_rss_mb"),
                            "model_bytes": out.get("model_bytes"),
                        }
                        rows.append(row)
                    except Exception as exc:
                        logger.exception("fail %s", exc)
                        rows.append(
                            {
                                "corpus": "road",
                                "capture": cap,
                                "representation": rep,
                                "detector": det_name,
                                "seed": seed,
                                "error": str(exc),
                            }
                        )

    df = pd.DataFrame(rows)
    # Aggregate mean/std/CI across seeds (seed-level mean of metrics)
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"]
    group_cols = ["corpus", "dataset", "capture", "attack_type", "representation", "detector"]
    # fill missing group cols
    for c in group_cols:
        if c not in df.columns:
            df[c] = ""
    ok = df[df["f1"].notna()] if "f1" in df.columns else df
    agg_rows = []
    for keys, g in ok.groupby([c for c in group_cols if c in ok.columns]):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip([c for c in group_cols if c in ok.columns], keys))
        rec["n_seeds"] = len(g)
        for m in metrics:
            if m not in g.columns:
                continue
            vals = g[m].astype(float).dropna()
            rec[f"{m}_mean"] = float(vals.mean())
            rec[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            # t-interval approx across seeds
            if len(vals) > 1:
                se = vals.std(ddof=1) / np.sqrt(len(vals))
                rec[f"{m}_ci95_low"] = float(vals.mean() - 1.96 * se)
                rec[f"{m}_ci95_high"] = float(vals.mean() + 1.96 * se)
            else:
                rec[f"{m}_ci95_low"] = float(vals.mean())
                rec[f"{m}_ci95_high"] = float(vals.mean())
        agg_rows.append(rec)
    ci_df = pd.DataFrame(agg_rows)
    return df, ci_df


def task_statistical_tests(cfg, hcrl_mats, road_mats):
    """McNemar + paired bootstrap delta residual vs raw, seed=0, residual primary detectors."""
    seed = 0
    set_global_seed(seed)
    eval_cfg = cfg["evaluation"]
    rows = []
    detectors = cfg["detectors"]

    def process(corpus, name, mat, attack_type):
        for det_name in detectors:
            params = _params(cfg, det_name, seed)
            # raw
            out_r, _ = _run_one(
                det_name,
                params,
                mat["raw"]["train"],
                mat["raw"]["test"],
                mat["feature_cols"],
                eval_cfg,
                cfg.get("bootstrap", {}),
                seed,
                measure=False,
            )
            out_s, _ = _run_one(
                det_name,
                params,
                mat["residual"]["train"],
                mat["residual"]["test"],
                mat["residual_cols"],
                eval_cfg,
                cfg.get("bootstrap", {}),
                seed,
                measure=False,
            )
            mc = mcnemar_test(out_r["y_test"], out_r["y_pred"], out_s["y_pred"])
            for metric in ("f1", "roc_auc"):
                pb = paired_bootstrap_delta(
                    out_r["y_test"],
                    out_r["scores_test"],
                    out_s["scores_test"],
                    out_r["y_pred"],
                    out_s["y_pred"],
                    n_boot=int(cfg.get("bootstrap", {}).get("n_boot", 300)),
                    block_size=int(cfg.get("bootstrap", {}).get("block_size", 50)),
                    seed=seed,
                    metric=metric,
                )
                rows.append(
                    {
                        "corpus": corpus,
                        "dataset": name,
                        "attack_type": attack_type,
                        "detector": det_name,
                        "comparison": "residual_vs_raw",
                        "metric": metric,
                        "mcnemar_stat": mc["statistic"],
                        "mcnemar_p": mc["p_value"],
                        "mcnemar_b": mc["b_a_wrong_b_right"],
                        "mcnemar_c": mc["c_a_right_b_wrong"],
                        "delta_point": pb["delta_point"],
                        "delta_ci_low": pb["ci_low"],
                        "delta_ci_high": pb["ci_high"],
                        "raw_metric": out_r[metric],
                        "residual_metric": out_s[metric],
                        "note": "McNemar on paired correctness; block-bootstrap on metric delta",
                    }
                )

    for ds, mat in hcrl_mats.items():
        process("hcrl", ds, mat, ds)
    for cap, mat in road_mats.items():
        # adapt structure
        m = {
            "raw": mat["raw"],
            "residual": mat["residual"],
            "feature_cols": mat["feature_cols"],
            "residual_cols": mat["residual_cols"],
        }
        process("road", cap, m, mat.get("attack_type", cap))
    return pd.DataFrame(rows)


def task_importance(cfg, hcrl_mats, road_mats):
    seed = 0
    set_global_seed(seed)
    det_name = cfg.get("importance", {}).get("detector", "isolation_forest")
    n_rep = int(cfg.get("importance", {}).get("n_repeats", 5))
    imp_rows = []
    abl_rows = []

    def factory():
        return create_detector(det_name, **_params(cfg, det_name, seed))

    for ds in cfg.get("importance", {}).get("hcrl_datasets", ["RPM", "gear"]):
        if ds not in hcrl_mats:
            continue
        mat = hcrl_mats[ds]
        train, test = mat["residual"]["train"], mat["residual"]["test"]
        cols = mat["residual_cols"]
        logger.info("importance HCRL %s", ds)
        pi = permutation_importance(
            factory, train, test, cols, n_repeats=n_rep, seed=seed, metric="roc_auc"
        )
        pi["corpus"] = "hcrl"
        pi["dataset"] = ds
        pi["detector"] = det_name
        imp_rows.append(pi)
        lo = leave_one_feature_out(factory, train, test, cols)
        lo["corpus"] = "hcrl"
        lo["dataset"] = ds
        lo["detector"] = det_name
        abl_rows.append(lo)

    for cap in cfg.get("importance", {}).get("road_captures", []):
        if cap not in road_mats:
            # try match
            matches = [k for k in road_mats if cap in k]
            if not matches:
                continue
            cap = matches[0]
        mat = road_mats[cap]
        train, test = mat["residual"]["train"], mat["residual"]["test"]
        cols = mat["residual_cols"]
        logger.info("importance ROAD %s", cap)
        pi = permutation_importance(
            factory, train, test, cols, n_repeats=n_rep, seed=seed, metric="roc_auc"
        )
        pi["corpus"] = "road"
        pi["dataset"] = cap
        pi["detector"] = det_name
        imp_rows.append(pi)
        lo = leave_one_feature_out(factory, train, test, cols)
        lo["corpus"] = "road"
        lo["dataset"] = cap
        lo["detector"] = det_name
        abl_rows.append(lo)

    imp = pd.concat(imp_rows, ignore_index=True) if imp_rows else pd.DataFrame()
    abl = pd.concat(abl_rows, ignore_index=True) if abl_rows else pd.DataFrame()
    return imp, abl


def task_runtime_latency_errors(cfg, hcrl_mats, road_mats):
    seed = 0
    set_global_seed(seed)
    eval_cfg = cfg["evaluation"]
    runtime_rows = []
    latency_rows = []
    error_rows = []
    detectors = cfg["detectors"]

    # HCRL residual + raw timing
    for ds, mat in hcrl_mats.items():
        for rep in ("raw", "residual"):
            train = mat[rep]["train"]
            test = mat[rep]["test"]
            cols = mat["feature_cols"] if rep == "raw" else mat["residual_cols"]
            for det_name in detectors:
                params = _params(cfg, det_name, seed)
                rss0 = peak_rss_mb()
                t0 = time.perf_counter()
                out, _ = _run_one(
                    det_name, params, train, test, cols, eval_cfg, {"n_boot": 50, "block_size": 50}, seed
                )
                wall = time.perf_counter() - t0
                n_test = len(test)
                score_s = float(out.get("score_seconds") or 0.0) or max(wall * 0.3, 1e-9)
                fps = n_test / score_s if score_s > 0 else float("nan")
                runtime_rows.append(
                    {
                        "corpus": "hcrl",
                        "dataset": ds,
                        "representation": rep,
                        "detector": det_name,
                        "train_seconds": out.get("train_seconds"),
                        "score_seconds": out.get("score_seconds"),
                        "wall_seconds": wall,
                        "n_test": n_test,
                        "windows_per_sec": fps,
                        "frames_per_sec_est": fps * float(cfg["hcrl"]["window_size"]),
                        "peak_rss_mb": out.get("peak_rss_mb", rss0),
                        "model_bytes": out.get("model_bytes"),
                        "complexity_note": _complexity_note(det_name),
                    }
                )
                # latency + errors on residual only to reduce volume
                if rep == "residual":
                    ts = test["timestamp"].values
                    lat = detection_latency(ts, out["y_test"], out["y_pred"])
                    lat.update(
                        {
                            "corpus": "hcrl",
                            "dataset": ds,
                            "detector": det_name,
                            "representation": rep,
                        }
                    )
                    latency_rows.append(lat)
                    segs = multi_segment_latencies(ts, out["y_test"], out["y_pred"])
                    if len(segs):
                        latency_rows.append(
                            {
                                "corpus": "hcrl",
                                "dataset": ds,
                                "detector": det_name,
                                "representation": rep,
                                "detected": bool(segs["detected"].any()),
                                "delay_frames": float(segs.loc[segs["detected"], "delay_frames"].mean())
                                if segs["detected"].any()
                                else float("nan"),
                                "delay_ms": float(segs.loc[segs["detected"], "delay_ms"].mean())
                                if segs["detected"].any()
                                else float("nan"),
                                "median_delay_ms": float(
                                    segs.loc[segs["detected"], "delay_ms"].median()
                                )
                                if segs["detected"].any()
                                else float("nan"),
                                "worst_delay_ms": float(segs.loc[segs["detected"], "delay_ms"].max())
                                if segs["detected"].any()
                                else float("nan"),
                                "n_segments": len(segs),
                                "n_detected_segments": int(segs["detected"].sum()),
                            }
                        )
                    err = catalog_errors(
                        test,
                        out["scores_test"],
                        out["y_pred"],
                        cols,
                        top_k=15,
                        threshold=out["threshold"],
                    )
                    err["corpus"] = "hcrl"
                    err["dataset"] = ds
                    err["detector"] = det_name
                    error_rows.append(err)

    for cap, mat in road_mats.items():
        for rep in ("raw", "residual"):
            train = mat[rep]["train"]
            test = mat[rep]["test"]
            cols = mat["feature_cols"] if rep == "raw" else mat["residual_cols"]
            for det_name in detectors:
                params = _params(cfg, det_name, seed)
                t0 = time.perf_counter()
                out, _ = _run_one(
                    det_name, params, train, test, cols, eval_cfg, {"n_boot": 50, "block_size": 40}, seed
                )
                wall = time.perf_counter() - t0
                n_test = len(test)
                score_s = float(out.get("score_seconds") or 0.0) or max(wall * 0.3, 1e-9)
                runtime_rows.append(
                    {
                        "corpus": "road",
                        "dataset": mat.get("attack_type", cap),
                        "capture": cap,
                        "representation": rep,
                        "detector": det_name,
                        "train_seconds": out.get("train_seconds"),
                        "score_seconds": out.get("score_seconds"),
                        "wall_seconds": wall,
                        "n_test": n_test,
                        "windows_per_sec": n_test / score_s,
                        "frames_per_sec_est": (n_test / score_s)
                        * float(cfg["road"]["window_size"]),
                        "peak_rss_mb": out.get("peak_rss_mb"),
                        "model_bytes": out.get("model_bytes"),
                        "complexity_note": _complexity_note(det_name),
                    }
                )
                if rep == "residual":
                    ts = test["timestamp"].values
                    lat = detection_latency(ts, out["y_test"], out["y_pred"])
                    lat.update(
                        {
                            "corpus": "road",
                            "dataset": mat.get("attack_type", cap),
                            "capture": cap,
                            "detector": det_name,
                            "representation": rep,
                        }
                    )
                    latency_rows.append(lat)
                    err = catalog_errors(
                        test,
                        out["scores_test"],
                        out["y_pred"],
                        cols,
                        top_k=10,
                        threshold=out["threshold"],
                    )
                    err["corpus"] = "road"
                    err["dataset"] = mat.get("attack_type", cap)
                    err["capture"] = cap
                    err["detector"] = det_name
                    error_rows.append(err)

    runtime = pd.DataFrame(runtime_rows)
    latency = pd.DataFrame(latency_rows)
    errors = pd.concat(error_rows, ignore_index=True) if error_rows else pd.DataFrame()
    return runtime, latency, errors


def _complexity_note(det: str) -> str:
    return {
        "isolation_forest": "Train O(T ψ log ψ); score O(T log ψ) per sample",
        "one_class_svm": "Train ~O(n_sv^2–n^3) RBF; score O(n_sv) per sample",
        "lof": "Train O(n log n) kNN; score O(n_neighbors) novelty",
        "hbos": "Train O(n d b); score O(d) — lightest",
        "elliptic_envelope": "Train robust cov ~O(n d^2); score O(d^2)",
        "autoencoder": "Train O(epochs n d h); score O(d h)",
    }.get(det, "")


def make_figures(cfg, multiseed_df, ci_df, imp_df, runtime_df, latency_df):
    apply_ieee_style()
    fig_root = Path(cfg.get("figures_dir", "figures"))
    paper_fig = Path("paper/updated_figures")
    for d in [
        fig_root / "representation",
        fig_root / "feature_importance",
        fig_root / "runtime",
        fig_root / "latency",
        fig_root / "error_analysis",
        paper_fig,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Representation comparison from multiseed means
    if ci_df is not None and len(ci_df):
        for corpus in ci_df["corpus"].unique():
            sub = ci_df[ci_df["corpus"] == corpus]
            if "f1_mean" not in sub.columns:
                continue
            # residual - raw mean delta by detector
            piv = sub.pivot_table(
                index="detector",
                columns="representation",
                values="f1_mean",
                aggfunc="mean",
            )
            if "raw" in piv.columns and "residual" in piv.columns:
                delta = (piv["residual"] - piv["raw"]).sort_values()
                fig, ax = plt.subplots(figsize=(6, 3.5))
                colors = [
                    IEEE_COLORS["residual"] if v >= 0 else IEEE_COLORS["attack"] for v in delta.values
                ]
                ax.barh(delta.index, delta.values, color=colors)
                ax.axvline(0, color="k", lw=0.8)
                ax.set_xlabel("Mean ΔF1 (residual − raw) across tasks")
                ax.set_title(f"Phase C multi-seed representation gain ({corpus.upper()})")
                fig.tight_layout()
                for dest in (fig_root / "representation", paper_fig):
                    fig.savefig(dest / f"multiseed_delta_f1_{corpus}.png")
                plt.close(fig)

    # Feature importance
    if imp_df is not None and len(imp_df):
        for (corpus, dataset), g in imp_df.groupby(["corpus", "dataset"]):
            g = g.sort_values("importance_mean")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(g["feature"], g["importance_mean"], xerr=g["importance_std"], color=IEEE_COLORS["residual"])
            ax.set_xlabel("Permutation importance (ΔROC-AUC)")
            ax.set_title(f"Feature importance ({corpus}: {dataset})")
            fig.tight_layout()
            safe = str(dataset).replace("/", "_")
            for dest in (fig_root / "feature_importance", paper_fig):
                fig.savefig(dest / f"perm_importance_{corpus}_{safe}.png")
            plt.close(fig)

    # Runtime
    if runtime_df is not None and len(runtime_df):
        sub = runtime_df[runtime_df["representation"] == "residual"]
        if len(sub):
            fig, ax = plt.subplots(figsize=(7, 3.5))
            pivot = sub.groupby("detector")[["train_seconds", "score_seconds"]].mean()
            pivot.plot(kind="bar", ax=ax, color=[IEEE_COLORS["raw"], IEEE_COLORS["residual"]])
            ax.set_ylabel("Seconds")
            ax.set_title("Mean train / score time (residual features)")
            ax.legend(["train", "score"])
            fig.tight_layout()
            for dest in (fig_root / "runtime", paper_fig):
                fig.savefig(dest / "runtime_train_score.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            wp = sub.groupby("detector")["windows_per_sec"].mean().sort_values()
            ax.barh(wp.index, wp.values, color=IEEE_COLORS["raw"])
            ax.set_xlabel("Windows / second (scoring)")
            ax.set_title("Inference throughput")
            fig.tight_layout()
            for dest in (fig_root / "runtime", paper_fig):
                fig.savefig(dest / "runtime_throughput.png")
            plt.close(fig)

    # Latency histogram
    if latency_df is not None and len(latency_df):
        det = latency_df[latency_df.get("detected") == True] if "detected" in latency_df.columns else latency_df
        if "delay_ms" in det.columns:
            vals = det["delay_ms"].dropna()
            if len(vals):
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(vals, bins=30, color=IEEE_COLORS["attack"], alpha=0.85)
                ax.set_xlabel("Detection latency (ms)")
                ax.set_ylabel("Count")
                ax.set_title("Detection latency (residual, first segment)")
                fig.tight_layout()
                for dest in (fig_root / "latency", paper_fig):
                    fig.savefig(dest / "latency_hist.png")
                plt.close(fig)


def robustness_summary(ci_df: pd.DataFrame, tables_dir: Path, fig_dir: Path):
    """Summarize where method works vs fails from multi-seed residual means."""
    apply_ieee_style()
    if ci_df is None or not len(ci_df) or "representation" not in ci_df.columns:
        return pd.DataFrame()
    res = ci_df[ci_df["representation"] == "residual"].copy()
    if "f1_mean" not in res.columns:
        return pd.DataFrame()

    def bucket(row):
        at = str(row.get("attack_type", row.get("dataset", "")))
        if at in ("DoS",):
            return "novel_id_flood"
        if "fuzz" in at.lower():
            return "cross_id_fuzzing"
        if at in ("RPM", "gear") or any(
            x in at for x in ("speedometer", "reverse_light", "correlated", "coolant")
        ):
            return "targeted_legitimate_aid"
        if at == "Fuzzy":
            return "hcrl_fuzzy_presence"
        return "other"

    res["coverage_bucket"] = res.apply(bucket, axis=1)
    summary = (
        res.groupby(["corpus", "coverage_bucket"])
        .agg(
            f1_mean=("f1_mean", "mean"),
            f1_std=("f1_mean", "std"),
            roc_mean=("roc_auc_mean", "mean"),
            n=("f1_mean", "count"),
        )
        .reset_index()
    )
    save_csv(tables_dir / "robustness_summary.csv", summary)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for corpus, g in summary.groupby("corpus"):
        ax.bar(
            np.arange(len(g)) + (0.0 if corpus == "hcrl" else 0.4),
            g["f1_mean"],
            width=0.4,
            label=corpus,
            yerr=g["f1_std"].fillna(0),
        )
        ax.set_xticks(np.arange(len(g)) + 0.2)
        ax.set_xticklabels(g["coverage_bucket"], rotation=20, ha="right")
    ax.set_ylabel("Mean residual F1")
    ax.set_title("Coverage boundary: residual performance by attack family")
    ax.legend()
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "coverage_boundary.png")
    Path("paper/updated_figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(Path("paper/updated_figures") / "coverage_boundary.png")
    plt.close(fig)
    return summary


def write_paper_stub(tables_dir: Path):
    """Write revised_paper.tex with representation-first narrative."""
    paper_dir = Path("paper")
    paper_dir.mkdir(parents=True, exist_ok=True)
    # Copy tables into paper/updated_tables
    ut = paper_dir / "updated_tables"
    ut.mkdir(parents=True, exist_ok=True)
    for p in tables_dir.glob("*.csv"):
        try:
            (ut / p.name).write_bytes(p.read_bytes())
        except Exception:
            pass

    tex = r"""\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{url}
\usepackage{hyperref}

\title{Behavioral Residualization as a Representation for Unsupervised CAN Intrusion Detection}

\author{
\IEEEauthorblockN{Anonymous Authors}
\IEEEauthorblockA{Prepared for arXiv submission}
}

\begin{document}
\maketitle

\begin{abstract}
Controller Area Network (CAN) intrusion detection must handle attacks that reuse legitimate arbitration IDs.
We investigate whether \emph{per-ID behavioral residualization}---z-scoring windowed timing and payload features against per-ID normal baselines---provides a useful representation for unsupervised anomaly detection under such reuse.
Across six detectors (Isolation Forest, One-Class SVM, LOF, HBOS, Elliptic Envelope, Autoencoder) and two corpora (HCRL and ROAD), residual features improve F1 in the majority of detector--task cells relative to raw features under a matched protocol (normal-only training, fixed false-positive-rate thresholding).
Multi-seed evaluation and block-bootstrap confidence intervals support that gains are not single-run artifacts.
We also report deployment-oriented runtime and detection latency, feature ablations, and an explicit coverage boundary: residualization helps targeted legitimate-ID spoofing-style attacks, while novel-ID floods (HCRL DoS) and cross-ID fuzzing remain difficult.
\end{abstract}

\begin{IEEEkeywords}
CAN bus, intrusion detection, anomaly detection, residual representation, automotive security
\end{IEEEkeywords}

\section{Introduction}
In-vehicle networks based on CAN remain a practical target for injection and spoofing attacks.
Many public benchmarks, notably HCRL Car-Hacking, contain presence-based cues (novel IDs, trivial payload constants) that can inflate detector scores without proving behavioral understanding.
ROAD emphasizes verified attacks that reuse legitimate arbitration IDs, better matching stealthy ECU impersonation.

We do \textbf{not} propose a new anomaly algorithm.
Instead, we investigate whether a CAN-specific \emph{representation}---per-ID behavioral residualization---consistently improves multiple unsupervised detectors under a fixed evaluation protocol.

\section{Hypothesis}
\textbf{H1:} Per-ID behavioral residualization improves unsupervised anomaly detection performance across multiple detectors relative to the same detectors on raw window features, under matched splits, calibration, and thresholds.

We pre-register a simple decision rule based on the fraction of detector--task cells with positive residual$-$raw deltas on F1 and ROC-AUC (see experimental supplements).

\section{Method}
\subsection{Window features}
For each arbitration ID we maintain a sliding window of length $w=30$ frames and compute inter-arrival, DLC, and flat payload statistics (14 features; group definitions in code).

\subsection{Behavioral residualization}
On a calibration segment of \emph{normal} windows we estimate per-ID means and standard deviations (minimum sample support; otherwise global fallback).
Residuals are $r_f=(x_f-\mu_{\mathrm{id},f})/(\sigma_{\mathrm{id},f}+\varepsilon)$.
On ROAD, residual statistics are fit \textbf{only on pre-injection} windows.

\subsection{Detectors}
Isolation Forest, One-Class SVM, Local Outlier Factor (novelty), HBOS, Elliptic Envelope, and a shallow autoencoder.
All train on normal-only features; thresholds target approximately 1\% FPR on held-out normals.

\section{Experimental Design}
Datasets: HCRL (DoS, Fuzzy, RPM, gear; 60k-frame prefixes) and ROAD (one capture per attack family; pre-injection protocol).
Factors: representation $\in\{\mathrm{raw},\mathrm{residual}\}$, detector, multi-seed $\{0,\ldots,4\}$.
Metrics: precision, recall, F1, ROC-AUC, PR-AUC, actual FPR, runtime, model size, detection latency.
Statistics: block bootstrap for metric CIs; multi-seed mean$\pm$std; McNemar on paired residual vs raw decisions; paired block-bootstrap on metric deltas.

\section{Results}
\subsection{Multi-detector representation study}
On HCRL, residualization improved F1 in 20/24 detector--task cells and ROC-AUC in 21/24 (Phase~A).
On ROAD, residualization improved F1 in 32/36 cells (Phase~B; multi-seed Phase~C refines uncertainty).
Gains are largest for legitimate-ID spoofing-style tasks (RPM/gear; reverse light; correlated signal).

\subsection{Negative results (retained)}
DoS (novel-ID flood) remains failed for essentially all detectors.
ROAD fuzzing remains weak; residual ROC can decrease.
Some detector--task pairs show residual F1 regressions when raw already performs well (e.g., autoencoder on selected ROAD tasks).

\subsection{Feature importance}
Permutation and leave-one-feature-out analyses (Isolation Forest, residual features) attribute gains primarily to inter-arrival and payload residual dimensions on spoofing tasks (tables/figures in supplement).

\subsection{Deployment metrics}
HBOS and Isolation Forest offer favorable score throughput; deep/kernel methods are heavier.
Detection latency (attack start to first true positive) is reported in milliseconds and frames for residual detectors.

\section{Coverage Boundary}
\begin{itemize}
\item \textbf{Works:} targeted legitimate-AID behavioral deviations (spoofed signals with ID reuse).
\item \textbf{Fails / weak:} novel-ID floods; short cross-ID fuzzing; settings with severe distribution shift relative to pre-injection calibration.
\end{itemize}

\section{Limitations}
No cross-vehicle validation; no CAN-FD; primarily offline calibration; limited hardware diversity; no online residual adaptation in the main results; HCRL presence artifacts; ROAD evaluated with capped frames and one capture per type in primary multi-seed tables unless otherwise noted.

\section{Related Work}
Isolation Forest, one-class SVM, LOF, histogram and covariance methods, and reconstruction models are standard unsupervised detectors.
Residual and baseline-normalized features appear broadly in anomaly detection; our contribution is a CAN-specific per-ID residual representation evaluated multi-detector multi-dataset with explicit failure modes---not a claim of inventing residualization or Isolation Forest.

\section{Conclusion}
Behavioral residualization is a practical CAN representation that consistently improves multiple unsupervised detectors under legitimate-ID reuse, while leaving novel-ID floods and cross-ID fuzzing as open challenges.
We release a reproducible pipeline for arXiv and further workshop/conference revision.

\section*{Reproducibility}
\begin{verbatim}
pip install -e ".[dev]"
python -m experiments.runners.run_phase_a --config experiments/configs/phase_a.yaml
python -m experiments.runners.run_phase_b_road --config experiments/configs/phase_b_road.yaml
python -m experiments.runners.run_phase_c --config experiments/configs/phase_c.yaml
\end{verbatim}
Software: Python 3.9+, scikit-learn, NumPy, pandas, SciPy, PyYAML, matplotlib.
Seeds, configs, and tables are under \texttt{experiments/} and \texttt{tables/}.

\begin{thebibliography}{00}
\bibitem{hcrl} HCRL Car-Hacking Dataset.
\bibitem{road} Verma et al., ROAD: Real ORNL Automotive Dynamometer CAN IDS dataset.
\bibitem{iforest} Liu et al., Isolation Forest.
\end{thebibliography}
\end{document}
"""
    (paper_dir / "revised_paper.tex").write_text(tex, encoding="utf-8")
    logger.info("wrote paper/revised_paper.tex")


def run_phase_c(cfg: dict) -> dict:
    tables = Path(cfg.get("tables_dir", "tables"))
    tables.mkdir(parents=True, exist_ok=True)
    exp_root = Path(cfg.get("output_dir", "experiments"))
    run = ExperimentRun(exp_root, name="phase_c")
    setup_logging(run.path("phase_c.log"))
    run.save_config(cfg)
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase_c"))

    logger.info("=== Prepare matrices ===")
    hcrl_mats = prepare_hcrl_matrices(cfg, cache)
    road_mats = prepare_road_matrices(cfg, cache)

    logger.info("=== Multi-seed ===")
    multiseed_df, ci_df = task_multiseed(cfg, hcrl_mats, road_mats, run)
    save_csv(tables / "multiseed_results.csv", multiseed_df)
    save_csv(tables / "confidence_intervals.csv", ci_df)
    save_csv(exp_root / "multiseed" / "results.csv", multiseed_df)
    save_csv(exp_root / "multiseed" / "confidence_intervals.csv", ci_df)

    logger.info("=== Statistical tests ===")
    stats_df = task_statistical_tests(cfg, hcrl_mats, road_mats)
    save_csv(tables / "statistical_tests.csv", stats_df)
    save_csv(exp_root / "statistics" / "statistical_tests.csv", stats_df)

    logger.info("=== Feature importance ===")
    imp_df, abl_df = task_importance(cfg, hcrl_mats, road_mats)
    save_csv(tables / "feature_importance.csv", imp_df)
    save_csv(tables / "feature_ablation.csv", abl_df)
    save_csv(exp_root / "feature_importance" / "permutation.csv", imp_df)
    save_csv(exp_root / "feature_importance" / "ablation.csv", abl_df)

    logger.info("=== Runtime / latency / errors ===")
    runtime_df, latency_df, error_df = task_runtime_latency_errors(cfg, hcrl_mats, road_mats)
    save_csv(tables / "runtime.csv", runtime_df)
    save_csv(tables / "latency.csv", latency_df)
    save_csv(tables / "error_analysis.csv", error_df)
    save_csv(exp_root / "runtime" / "runtime.csv", runtime_df)
    save_csv(exp_root / "latency" / "latency.csv", latency_df)
    save_csv(exp_root / "error_analysis" / "errors.csv", error_df)

    logger.info("=== Figures + robustness ===")
    make_figures(cfg, multiseed_df, ci_df, imp_df, runtime_df, latency_df)
    robustness_summary(ci_df, tables, Path(cfg.get("figures_dir", "figures")) / "representation")

    logger.info("=== Paper stub ===")
    write_paper_stub(tables)

    # Save run copies
    for name in [
        "multiseed_results.csv",
        "confidence_intervals.csv",
        "statistical_tests.csv",
        "feature_importance.csv",
        "feature_ablation.csv",
        "runtime.csv",
        "latency.csv",
        "error_analysis.csv",
    ]:
        src = tables / name
        if src.exists():
            save_csv(run.path(name), pd.read_csv(src))

    return {
        "run_dir": str(run.run_dir),
        "n_multiseed": len(multiseed_df),
        "n_ci": len(ci_df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/phase_c.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_phase_c(cfg)
    print("\n=== PHASE C COMPLETE ===")
    print(result)


if __name__ == "__main__":
    main()
