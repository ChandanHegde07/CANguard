"""Phase 2 hybrid experiment: global-only + PIRD/global hybrid detection.

Invoked through the existing Phase 2 runner::

    python -m experiments.runners.run_phase2 \\
        --config experiments/configs/phase2_hybrid.yaml --mode hybrid

Scientific protocol
-------------------
* The PIRD branch is exactly the frozen Phase 1/2 pipeline (per-ID z-score +
  Isolation Forest, threshold at 1% FPR on held-out train normals).
* The global branch is an independently-defined, causal, window-causal
  representation of bus/ID-population/cross-ID behaviour, normalized globally
  and scored by the *same* detector abstraction.
* Both branches are calibrated on the same temporal split and frozen before any
  attack is generated. No test/attack statistics enter calibration.
* Adaptive attacks are applied unchanged (reusing ``canguard.attacks``), and
  every adaptive result is reported with a matched unmodified control and a
  two-proportion significance test (reusing ``run_phase2._two_proportion_z``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from canguard.attacks import (
    AttackParameters,
    apply_alpha_series,
    apply_fixed_attack,
    gradual_alphas,
    resolve_target_features,
)
from canguard.evaluation import compute_metrics, train_anomaly_detector
from canguard.evaluation.boundary import detection_probability, summarize_boundary
from canguard.exp import FeatureCache, save_csv, save_json, set_global_seed
from canguard.exp.matrix import build_window_table, resolve_data_path
from canguard.features import (
    ALL_GLOBAL_FEATURES,
    GLOBAL_ABLATIONS,
    build_global_features,
    fit_global_stats,
    fit_per_id_stats,
    temporal_split,
    transform_global,
    transform_residuals,
)
from canguard.hybrid import (
    DEFAULT_LAMBDAS,
    fit_score_normalizer,
    or_predict,
    threshold_at_fpr,
)

from .run_phase2 import (
    _experiment_meta,
    _feature_cols,
    _make_detector,
    _two_proportion_z,
    select_legit_targets,
)

HYBRID_FPR_TARGETS = (0.001, 0.01, 0.05)
PRIMARY_RULES = ("pird", "global", "or", "max", "weighted_0.5")


def _sig(ak: int, an: int, ck: int, cn: int, pval: float) -> bool:
    """Control-adjusted significance: attack alarm rate above control, p<0.05."""
    if not (an and cn):
        return False
    return bool((ak / an) > (ck / cn) and pval < 0.05)


# ---------------------------------------------------------------------------
# Branch construction
# ---------------------------------------------------------------------------
def _make_global_detector(cfg: dict):
    gcfg = cfg.get("global", {})
    if gcfg.get("detector"):
        return _make_detector({"detector": gcfg["detector"]})
    return _make_detector(cfg)


def _val_normal_scores(model, train_res_df, res_cols, val_fraction=0.2) -> np.ndarray:
    tn = train_res_df[train_res_df["is_attack"] == 0]
    n = len(tn)
    if n == 0:
        return np.zeros(0)
    n_val = max(1, int(n * val_fraction))
    val = tn.iloc[-n_val:] if n_val < n else tn
    return model.score_samples(val[res_cols].fillna(0).values)


def _global_refs(calib_norm: pd.DataFrame) -> tuple[set[str], dict[str, float]]:
    ids = set(calib_norm["can_id"].astype(str))
    freq = calib_norm["can_id"].astype(str).value_counts(normalize=True).to_dict()
    return ids, freq


def _build_global_table(wt: pd.DataFrame, refs, gcfg: dict) -> pd.DataFrame:
    ids, freq = refs
    return build_global_features(
        wt,
        window_frames=int(gcfg.get("window_frames", 200)),
        id_activity_frac=float(gcfg.get("id_activity_frac", 0.05)),
        calibration_ids=ids,
        calib_id_freq=freq,
    )


def _fit_global_branch(
    gcalib, gtrain, gtest, cols, cfg, method, val_fraction, fpr_target
) -> dict:
    gs = fit_global_stats(gcalib, cols, method=method)
    rtr = transform_global(gtrain, gs)
    rte = transform_global(gtest, gs)
    rcols = [c + "_gres" for c in cols]
    det = _make_global_detector(cfg)
    out = train_anomaly_detector(
        det,
        rtr,
        rte,
        rcols,
        val_holdout_fraction=val_fraction,
        fpr_target=fpr_target,
    )
    val = _val_normal_scores(out["model"], rtr, rcols, val_fraction)
    return {
        "stats": gs,
        "model": out["model"],
        "threshold": float(out["threshold"]),
        "res_cols": rcols,
        "val": val,
        "norm": fit_score_normalizer(val),
        "metrics": out,
    }


def _prepare_hybrid_dataset(name: str, cfg: dict, cache: FeatureCache) -> dict:
    feature_cols = _feature_cols(cfg)
    split = cfg["split"]
    eval_cfg = cfg.get("evaluation", {})
    val_fraction = eval_cfg.get("val_holdout_fraction", 0.2)
    fpr_target = eval_cfg.get("fpr_target", 0.01)

    wt = build_window_table(
        resolve_data_path(cfg.get("data_dir", "data"), name),
        window_size=int(cfg.get("window_size", 30)),
        sample_size=cfg.get("sample_size"),
        cache=cache,
    )
    calib, train, test = temporal_split(
        wt, split["calib_frac"], split["train_frac"], split["test_frac"]
    )
    calib_norm = calib[calib["is_attack"] == 0]

    # ---- PIRD branch (frozen) -------------------------------------------
    stats, gstats = fit_per_id_stats(calib, feature_cols)
    res_cols = [c + "_res" for c in feature_cols]
    res_train = transform_residuals(train, stats, gstats, feature_cols)
    res_test = transform_residuals(test, stats, gstats, feature_cols)
    out_p = train_anomaly_detector(
        _make_detector(cfg),
        res_train,
        res_test,
        res_cols,
        val_holdout_fraction=val_fraction,
        fpr_target=fpr_target,
    )
    pird = {
        "model": out_p["model"],
        "threshold": float(out_p["threshold"]),
        "val": _val_normal_scores(out_p["model"], res_train, res_cols, val_fraction),
        "metrics": out_p,
        "res_cols": res_cols,
        "stats": stats,
        "gstats": gstats,
    }
    pird["norm"] = fit_score_normalizer(pird["val"])

    # ---- Global branch ---------------------------------------------------
    gcfg = cfg.get("global", {})
    refs = _global_refs(calib_norm)
    gft = _build_global_table(wt, refs, gcfg)
    if len(gft) != len(wt):
        raise AssertionError("global feature table misaligned with window table")
    gcalib, gtrain, gtest = temporal_split(
        gft, split["calib_frac"], split["train_frac"], split["test_frac"]
    )
    global_cols = list(gcfg.get("feature_cols") or ALL_GLOBAL_FEATURES)
    method = gcfg.get("normalization", "robust")
    glob = _fit_global_branch(
        gcalib, gtrain, gtest, global_cols, cfg, method, val_fraction, fpr_target
    )

    # ---- Global ablation branches ---------------------------------------
    ablation: dict[str, dict] = {}
    if gcfg.get("run_ablation", True):
        for fam, cols in GLOBAL_ABLATIONS.items():
            ablation[fam] = _fit_global_branch(
                gcalib, gtrain, gtest, cols, cfg, method, val_fraction, fpr_target
            )

    return {
        "name": name,
        "feature_cols": feature_cols,
        "wt": wt,
        "calib": calib,
        "train": train,
        "test": test,
        "calib_norm": calib_norm,
        "refs": refs,
        "gft": gft,
        "gcalib": gcalib,
        "gtrain": gtrain,
        "gtest": gtest,
        "pird": pird,
        "global": glob,
        "ablation": ablation,
        "split": split,
        "eval_cfg": eval_cfg,
        "gcfg": gcfg,
        "method": method,
        "val_fraction": val_fraction,
        "fpr_target": fpr_target,
        "test_start": len(calib) + len(train),
    }


# ---------------------------------------------------------------------------
# Hybrid scoring helpers
# ---------------------------------------------------------------------------
def _hybrid_thresholds(val_pird, val_glob, pird_norm, glob_norm, fpr, lambdas) -> dict:
    zl = pird_norm.transform(val_pird)
    zg = glob_norm.transform(val_glob)
    th = {
        "pird": threshold_at_fpr(val_pird, fpr),
        "global": threshold_at_fpr(val_glob, fpr),
        "max": threshold_at_fpr(np.maximum(zl, zg), fpr),
    }
    for lam in lambdas:
        th[f"weighted_{lam:g}"] = threshold_at_fpr(lam * zl + (1.0 - lam) * zg, fpr)
    return th


def _hybrid_flags(pird_raw, glob_raw, pird_norm, glob_norm, th) -> dict:
    zl = pird_norm.transform(pird_raw)
    zg = glob_norm.transform(glob_raw)
    pl = pird_raw >= th["pird"]
    pg = glob_raw >= th["global"]
    out = {
        "pird": pl,
        "global": pg,
        "or": or_predict(pl, pg),
        "max": np.maximum(zl, zg) >= th["max"],
    }
    for k, v in th.items():
        if k.startswith("weighted_"):
            lam = float(k.split("_")[1])
            out[k] = (lam * zl + (1.0 - lam) * zg) >= v
    return out


# ---------------------------------------------------------------------------
# Conventional-attack evaluation + fixed-FPR + false alarms
# ---------------------------------------------------------------------------
def _conventional_tables(prep: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    pird, glob = prep["pird"], prep["global"]
    y = prep["gtest"]["is_attack"].to_numpy(dtype=int)
    ts = prep["gtest"]["timestamp"].to_numpy(dtype=float)
    duration_h = max((ts[-1] - ts[0]) / 3600.0, 1e-9) if len(ts) else 1e-9
    pird_raw = pird["metrics"]["scores_test"]
    glob_raw = glob["metrics"]["scores_test"]
    zl = pird["norm"].transform(pird_raw)
    zg = glob["norm"].transform(glob_raw)
    lambdas = list(cfg.get("hybrid", {}).get("lambdas", DEFAULT_LAMBDAS))

    metrics_rows: list[dict] = []
    fixed_rows: list[dict] = []
    for fpr in sorted({prep["fpr_target"], *HYBRID_FPR_TARGETS}):
        th = _hybrid_thresholds(pird["val"], glob["val"], pird["norm"], glob["norm"], fpr, lambdas)
        flags = _hybrid_flags(pird_raw, glob_raw, pird["norm"], glob["norm"], th)
        for rule, pred in flags.items():
            pred = pred.astype(int)
            if rule == "pird":
                score = pird_raw
            elif rule == "global":
                score = glob_raw
            elif rule == "max":
                score = np.maximum(zl, zg)
            elif rule.startswith("weighted_"):
                lam = float(rule.split("_")[1])
                score = lam * zl + (1.0 - lam) * zg
            else:  # or
                score = np.maximum(zl, zg)
            m = compute_metrics(y, pred, score)
            if rule == "or":
                m["roc_auc"] = float("nan")
                m["pr_auc"] = float("nan")
            fp = m["fp"]
            row = {
                "dataset": prep["name"],
                "rule": rule,
                "fpr_target": float(fpr),
                "threshold_pird": th["pird"],
                "threshold_global": th["global"],
                "n_test": int(len(y)),
                "n_attack": int(y.sum()),
                "actual_fpr": m["fpr"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "tp": m["tp"],
                "fp": fp,
                "fn": m["fn"],
                "tn": m["tn"],
                "false_alarms_per_hour": float(fp / duration_h),
                "false_alarms_per_window": float(fp / len(y)) if len(y) else float("nan"),
            }
            fixed_rows.append(row)
            if abs(fpr - prep["fpr_target"]) < 1e-12:
                metrics_rows.append(dict(row))
    return metrics_rows, fixed_rows


def _fpr_curve(prep: dict, cfg: dict) -> list[dict]:
    pird, glob = prep["pird"], prep["global"]
    y = prep["gtest"]["is_attack"].to_numpy(dtype=int)
    pird_raw = pird["metrics"]["scores_test"]
    glob_raw = glob["metrics"]["scores_test"]
    lambdas = list(cfg.get("hybrid", {}).get("lambdas", DEFAULT_LAMBDAS))
    rows = []
    for fpr in np.logspace(-3, -0.3, 15):
        th = _hybrid_thresholds(pird["val"], glob["val"], pird["norm"], glob["norm"],
                                float(fpr), lambdas)
        flags = _hybrid_flags(pird_raw, glob_raw, pird["norm"], glob["norm"], th)
        for rule, pred in flags.items():
            pred = pred.astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            rows.append({
                "dataset": prep["name"], "rule": rule, "target_fpr": float(fpr),
                "actual_fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
                "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
                "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            })
    return rows


def _complementarity(prep: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    pird, glob = prep["pird"], prep["global"]
    y = prep["gtest"]["is_attack"].to_numpy(dtype=int)
    pird_raw = pird["metrics"]["scores_test"]
    glob_raw = glob["metrics"]["scores_test"]
    fpr = prep["fpr_target"]
    th = _hybrid_thresholds(pird["val"], glob["val"], pird["norm"], glob["norm"], fpr,
                            list(cfg.get("hybrid", {}).get("lambdas", DEFAULT_LAMBDAS)))
    pl = pird_raw >= th["pird"]
    pg = glob_raw >= th["global"]
    rows = []
    subsets = (("all", np.ones(len(y), dtype=bool)), ("attack", y == 1), ("normal", y == 0))
    for subset, mask in subsets:
        pl_s, pg_s = pl[mask], pg[mask]
        rows.append(
            {
                "dataset": prep["name"],
                "subset": subset,
                "n": int(mask.sum()),
                "pird_only": int((pl_s & ~pg_s).sum()),
                "global_only": int((~pl_s & pg_s).sum()),
                "both": int((pl_s & pg_s).sum()),
                "neither": int((~pl_s & ~pg_s).sum()),
            }
        )
    corr_rows = []
    pa = pd.Series(pird_raw)
    ga = pd.Series(glob_raw)
    corr_rows.append({"dataset": prep["name"], "subset": "all",
                      "pearson": float(pa.corr(ga)),
                      "spearman": float(pa.corr(ga, method="spearman"))})
    if (y == 0).sum() > 2:
        pn = pd.Series(pird_raw[y == 0])
        gn = pd.Series(glob_raw[y == 0])
        corr_rows.append({"dataset": prep["name"], "subset": "normal",
                          "pearson": float(pn.corr(gn)),
                          "spearman": float(pn.corr(gn, method="spearman"))})
    return rows, corr_rows


# ---------------------------------------------------------------------------
# Adaptive attack sweep with global + hybrid
# ---------------------------------------------------------------------------
def _mutate_window_table(wt, positions, params, target, alphas, direction, feats, mode):
    """Return a copy of ``wt`` with only ``positions`` overwritten by attack values."""
    out = wt.copy()
    sub = out.iloc[positions].copy()
    sub = apply_alpha_series(params, sub, target, alphas, direction, feats, mode=mode)
    for f in feats:
        col = out.columns.get_loc(f)
        out.iloc[positions, col] = sub[f].to_numpy()
    return out


def _branch_scores(gft: pd.DataFrame, branch: dict) -> np.ndarray:
    """Score a global branch (main or ablation) on a prebuilt global table."""
    r = transform_global(gft, branch["stats"])
    return branch["model"].score_samples(r[branch["res_cols"]].fillna(0).values)


def _score_global_on_table(prep, wt_table) -> np.ndarray:
    g = _build_global_table(wt_table, prep["refs"], prep["gcfg"])
    return _branch_scores(g, prep["global"])


def run_hybrid_adaptive(cfg: dict, prepare, cache: FeatureCache) -> dict:
    atk = cfg["attack"]
    severities = [float(a) for a in atk["severities"]]
    directions = list(atk.get("directions", ["positive", "negative"]))
    modes = list(atk.get("attack_modes", ["additive"]))
    feature_spec = atk.get("target_features", "all_attackable")
    targeting = cfg.get("targeting", {})
    lambdas = list(cfg.get("hybrid", {}).get("lambdas", DEFAULT_LAMBDAS))
    ablation_sevs = set(
        float(a)
        for a in cfg.get("hybrid", {}).get("ablation_severities", [0.5, 1.0, 1.5])
    )

    rows: list[dict] = []
    for name in cfg["datasets"]:
        prep = prepare(name, cfg, cache)
        feature_cols = prep["feature_cols"]
        res_cols = prep["pird"]["res_cols"]
        test_start = prep["test_start"]
        wt = prep["wt"]
        ids = wt["can_id"].astype(str).values
        is_atk = wt["is_attack"].values
        n = len(wt)
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_start:] = True
        norm_mask = (is_atk == 0) & test_mask
        pird_raw_test = prep["pird"]["metrics"]["scores_test"]
        glob_raw_test = prep["global"]["metrics"]["scores_test"]

        params = AttackParameters.from_stats(
            prep["pird"]["stats"], prep["pird"]["gstats"], feature_cols,
            provenance="calib_normal", n_calib_normal_windows=len(prep["calib_norm"]),
        )
        targets = select_legit_targets(
            prep["calib"], prep["pird"]["stats"],
            top_k=int(targeting.get("top_k", 3)),
            min_calib_windows=int(targeting.get("min_calib_windows", 50)),
            max_calib_attack_frac=float(targeting.get("max_calib_attack_frac", 0.0)),
            explicit=targeting.get("target_ids"),
        )
        feats = resolve_target_features(feature_spec, feature_cols)
        # Clean (unmodified) scores for global ablation control, computed once.
        clean_abl_scores = {
            fam: _branch_scores(prep["gft"], br) for fam, br in prep["ablation"].items()
        }

        for target in targets:
            positions = np.flatnonzero(norm_mask & (ids == target))
            if len(positions) == 0:
                continue
            local_idx = positions - test_start
            ctrl_pird = pird_raw_test[local_idx]
            ctrl_glob = glob_raw_test[local_idx]
            ctrl_th = _hybrid_thresholds(
                prep["pird"]["val"], prep["global"]["val"],
                prep["pird"]["norm"], prep["global"]["norm"],
                prep["fpr_target"], lambdas,
            )
            ctrl_flags = _hybrid_flags(ctrl_pird, ctrl_glob,
                                       prep["pird"]["norm"], prep["global"]["norm"], ctrl_th)

            for mode in modes:
                for direction in directions:
                    for alpha in severities:
                        sub = wt.iloc[positions].copy()
                        sub = apply_fixed_attack(
                            params, sub, target, alpha, direction, feats, mode=mode
                        )
                        # PIRD attacked scores (linear, row-local).
                        res_att = transform_residuals(
                            sub, prep["pird"]["stats"], prep["pird"]["gstats"], feature_cols
                        )
                        pird_att = prep["pird"]["model"].score_samples(
                            res_att[res_cols].fillna(0).values
                        )
                        # Global attacked scores: rebuild the global table once
                        # and reuse it for the main branch and every ablation.
                        wt_att = _mutate_window_table(
                            wt,
                            positions,
                            params,
                            target,
                            np.full(len(positions), alpha),
                            direction,
                            feats,
                            mode,
                        )
                        gft_att = _build_global_table(wt_att, prep["refs"], prep["gcfg"])
                        glob_att = _branch_scores(gft_att, prep["global"])[positions]
                        att_flags = _hybrid_flags(
                            pird_att,
                            glob_att,
                            prep["pird"]["norm"],
                            prep["global"]["norm"],
                            ctrl_th,
                        )

                        def _record(rule, pred, cpred, rows=rows, name=name,
                                    target=target, mode=mode, direction=direction,
                                    alpha=alpha):
                            ak = int(pred.sum())
                            ck = int(cpred.sum())
                            an, cn = int(len(pred)), int(len(cpred))
                            z, pval = _two_proportion_z(ak, an, ck, cn)
                            sig = _sig(ak, an, ck, cn, pval)
                            rows.append({
                                "dataset": name, "target_id": target, "attack_mode": mode,
                                "direction": direction, "severity": alpha, "rule": rule,
                                "n_attack": an, "attack_detected": ak,
                                "attack_detection_rate": ak / an if an else float("nan"),
                                "control_rate": ck / cn if cn else float("nan"),
                                "alarm_lift": (ak / an - ck / cn) if an and cn else float("nan"),
                                "two_prop_z": z, "p_value_onesided": pval,
                                "detected_significant": sig,
                            })

                        for rule, pred in att_flags.items():
                            _record(rule, pred, ctrl_flags[rule])

                        if alpha in ablation_sevs:
                            for fam, br in prep["ablation"].items():
                                pred = _branch_scores(gft_att, br)[positions] >= br["threshold"]
                                cpred = clean_abl_scores[fam][positions] >= br["threshold"]
                                _record(f"global_{fam}", pred, cpred)
    return {"rows": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Gradual drift with global + hybrid
# ---------------------------------------------------------------------------
def run_hybrid_gradual(cfg: dict, prepare, cache: FeatureCache) -> dict:
    atk = cfg["attack"]
    drift = atk.get("drift", {})
    severities = [float(a) for a in atk["severities"]]
    directions = list(atk.get("directions", ["positive"]))
    modes = list(atk.get("attack_modes", ["additive"]))
    rates = list(drift.get("rates", ["fast", "medium", "slow"]))
    shape = drift.get("shape", "exponential")
    fractions = drift.get("fractions")
    horizon = int(drift.get("horizon_windows", 300))
    feature_spec = atk.get("target_features", "all_attackable")
    targeting = cfg.get("targeting", {})
    lambdas = list(cfg.get("hybrid", {}).get("lambdas", DEFAULT_LAMBDAS))
    save_traj = bool(drift.get("save_trajectories", True))

    result_rows: list[dict] = []
    traj_rows: list[dict] = []

    for name in cfg["datasets"]:
        prep = prepare(name, cfg, cache)
        feature_cols = prep["feature_cols"]
        res_cols = prep["pird"]["res_cols"]
        test_start = prep["test_start"]
        wt = prep["wt"]
        ids = wt["can_id"].astype(str).values
        is_atk = wt["is_attack"].values
        test_mask = np.zeros(len(wt), dtype=bool)
        test_mask[test_start:] = True
        norm_mask = (is_atk == 0) & test_mask
        params = AttackParameters.from_stats(
            prep["pird"]["stats"], prep["pird"]["gstats"], feature_cols,
            provenance="calib_normal", n_calib_normal_windows=len(prep["calib_norm"]),
        )
        targets = select_legit_targets(
            prep["calib"], prep["pird"]["stats"],
            top_k=int(targeting.get("top_k", 3)),
            min_calib_windows=int(targeting.get("min_calib_windows", 50)),
            max_calib_attack_frac=float(targeting.get("max_calib_attack_frac", 0.0)),
            explicit=targeting.get("target_ids"),
        )
        feats = resolve_target_features(feature_spec, feature_cols)
        th = _hybrid_thresholds(
            prep["pird"]["val"], prep["global"]["val"],
            prep["pird"]["norm"], prep["global"]["norm"], prep["fpr_target"], lambdas,
        )
        # Clean control scores for the whole test segment.
        pird_raw_test = prep["pird"]["metrics"]["scores_test"]
        glob_raw_test = prep["global"]["metrics"]["scores_test"]
        ctrl_flags_all = _hybrid_flags(pird_raw_test, glob_raw_test,
                                       prep["pird"]["norm"], prep["global"]["norm"], th)

        for target in targets:
            positions = np.flatnonzero(norm_mask & (ids == target))
            if len(positions) <= 5:
                continue
            if len(positions) <= horizon:
                block = positions
            else:
                start = (len(positions) - horizon) // 2
                block = positions[start:start + horizon]
            local_idx = block - test_start
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
                            sub = wt.iloc[block].copy()
                            sub = apply_alpha_series(
                                params, sub, target, alphas, direction, feats, mode=mode
                            )
                            res_att = transform_residuals(
                                sub, prep["pird"]["stats"], prep["pird"]["gstats"], feature_cols
                            )
                            pird_att = prep["pird"]["model"].score_samples(
                                res_att[res_cols].fillna(0).values
                            )
                            wt_att = _mutate_window_table(
                                wt, block, params, target, alphas, direction, feats, mode
                            )
                            glob_att = _score_global_on_table(prep, wt_att)[block]
                            att_flags = _hybrid_flags(
                                pird_att,
                                glob_att,
                                prep["pird"]["norm"],
                                prep["global"]["norm"],
                                th,
                            )
                            for rule, pred in att_flags.items():
                                cpred = ctrl_flags_all[rule][local_idx]
                                ak = int(pred.sum())
                                ck = int(cpred.sum())
                                an = int(len(pred))
                                cn = int(len(cpred))
                                z, pval = _two_proportion_z(ak, an, ck, cn)
                                detected = ak > 0
                                first = int(np.argmax(pred)) if detected else -1
                                sig = _sig(ak, an, ck, cn, pval)
                                result_rows.append({
                                    "dataset": name, "target_id": target, "attack_mode": mode,
                                    "direction": direction, "rate": rate, "shape": shape,
                                    "alpha_end": alpha_end, "rule": rule, "horizon": an,
                                    "detected_any": bool(detected),
                                    "attack_alarm_rate": ak / an if an else float("nan"),
                                    "control_alarm_rate": ck / cn if cn else float("nan"),
                                    "detected_significant": sig,
                                    "p_value_onesided": pval,
                                    "delay_windows": first,
                                    "max_residual_norm": float(np.max(np.linalg.norm(
                                        res_att[res_cols].to_numpy(dtype=float), axis=1))),
                                })
                                if (
                                    save_traj
                                    and mode == "additive"
                                    and rate == "slow"
                                    and direction == "positive"
                                ):
                                    zl = prep["pird"]["norm"].transform(pird_att)
                                    zg = prep["global"]["norm"].transform(glob_att)
                                    hz = np.maximum(zl, zg)
                                    for t in range(len(block)):
                                        traj_rows.append({
                                            "dataset": name, "target_id": target,
                                            "attack_mode": mode, "direction": direction,
                                            "rate": rate, "alpha_end": alpha_end,
                                            "t": t, "alpha": float(alphas[t]),
                                            "pird_score": float(pird_att[t]),
                                            "global_score": float(glob_att[t]),
                                            "hybrid_max_z": float(hz[t]),
                                            "flagged_pird": bool(att_flags["pird"][t]),
                                            "flagged_global": bool(att_flags["global"][t]),
                                            "flagged_max": bool(att_flags["max"][t]),
                                        })
    return {"results": pd.DataFrame(result_rows), "trajectories": pd.DataFrame(traj_rows)}


# ---------------------------------------------------------------------------
# Aggregation + plots
# ---------------------------------------------------------------------------
def _aggregate_adaptive(sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule, g in sweep.groupby("rule"):
        for sev, gs in g.groupby("severity"):
            n = int(gs["n_attack"].sum())
            k = int(gs["attack_detected"].sum())
            dp = detection_probability(np.concatenate([np.ones(k, int), np.zeros(n - k, int)]))
            rows.append({
                "rule": rule, "severity": float(sev), "n_attack": n, "n_detected": k,
                "detection_probability": dp["detection_probability"],
                "detection_prob_ci_low": dp["detection_prob_ci_low"],
                "detection_prob_ci_high": dp["detection_prob_ci_high"],
                "detection_probability_significant": float(gs["detected_significant"].mean()),
                "mean_control_rate": float(gs["control_rate"].mean(skipna=True)),
                "mean_alarm_lift": float(gs["alarm_lift"].mean(skipna=True)),
            })
    return pd.DataFrame(rows).sort_values(["rule", "severity"]).reset_index(drop=True)


def _aggregate_gradual(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rule, rate, ae), g in results.groupby(["rule", "rate", "alpha_end"]):
        det = g[g["detected_significant"]]
        rows.append({
            "rule": rule, "rate": rate, "alpha_end": float(ae), "n": int(len(g)),
            "detection_probability_any": float(g["detected_any"].mean()),
            "detection_significant": float(g["detected_significant"].mean()),
            "mean_attack_alarm_rate": float(g["attack_alarm_rate"].mean()),
            "mean_control_alarm_rate": float(g["control_alarm_rate"].mean()),
            "median_delay_windows_significant": (
                float(det["delay_windows"].median()) if len(det) else float("nan")
            ),
        })
    return pd.DataFrame(rows).sort_values(["rule", "rate", "alpha_end"]).reset_index(drop=True)


def run_hybrid(cfg: dict, out_root: Path, prepare=None) -> dict:
    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase2"))
    global_dir = out_root / "global"
    hybrid_dir = out_root / "hybrid"
    global_dir.mkdir(parents=True, exist_ok=True)
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    save_json(hybrid_dir / "config.json", cfg)

    if prepare is None:
        def prepare(name, c, ca):
            return _prepare_hybrid_dataset(name, c, ca)

    # ---- Conventional attacks: PIRD / global / hybrid --------------------
    conv_metrics, fixed_fpr, comp_rows, corr_rows, fpr_curve = [], [], [], [], []
    abl_conv: list[dict] = []
    norms = {}
    for name in cfg["datasets"]:
        prep = prepare(name, cfg, cache)
        m, f = _conventional_tables(prep, cfg)
        conv_metrics.extend(m)
        fixed_fpr.extend(f)
        c, corr = _complementarity(prep, cfg)
        comp_rows.extend(c)
        corr_rows.extend(corr)
        fpr_curve.extend(_fpr_curve(prep, cfg))
        # Global feature-family ablation on conventional attacks.
        y = prep["gtest"]["is_attack"].to_numpy(dtype=int)
        for fam, br in prep["ablation"].items():
            sc = br["metrics"]["scores_test"]
            pred = (sc >= br["threshold"]).astype(int)
            mm = compute_metrics(y, pred, sc)
            abl_conv.append({
                "dataset": name, "family": fam, "threshold": br["threshold"],
                "precision": mm["precision"], "recall": mm["recall"], "f1": mm["f1"],
                "actual_fpr": mm["fpr"], "roc_auc": mm["roc_auc"], "pr_auc": mm["pr_auc"],
            })
        # Save per-window scores for ROC/PR figures.
        pd.DataFrame({
            "y": prep["gtest"]["is_attack"].to_numpy(dtype=int),
            "pird_score": prep["pird"]["metrics"]["scores_test"],
            "global_score": prep["global"]["metrics"]["scores_test"],
        }).to_csv(hybrid_dir / f"scores_{name}.csv", index=False)
        norms[name] = {
            "local_center": prep["pird"]["norm"].center,
            "local_scale": prep["pird"]["norm"].scale,
            "global_center": prep["global"]["norm"].center,
            "global_scale": prep["global"]["norm"].scale,
        }
        print(f"[hybrid] conventional {name}: "
              + " | ".join(f"{r['rule']} F1={r['f1']:.3f} FPR={r['actual_fpr']:.3f}"
                           for r in m if r['rule'] in ('pird', 'global')))

    conv_df = pd.DataFrame(conv_metrics)
    fixed_df = pd.DataFrame(fixed_fpr)
    save_csv(hybrid_dir / "conventional_metrics.csv", conv_df)
    save_csv(hybrid_dir / "fixed_fpr.csv", fixed_df)
    save_csv(hybrid_dir / "fpr_recall_curve.csv", pd.DataFrame(fpr_curve))
    save_csv(hybrid_dir / "complementarity.csv", pd.DataFrame(comp_rows))
    save_csv(hybrid_dir / "score_correlation.csv", pd.DataFrame(corr_rows))
    save_csv(hybrid_dir / "ablation_conventional.csv", pd.DataFrame(abl_conv))
    save_json(hybrid_dir / "score_norms.json", norms)
    # Global-only conventional view.
    save_csv(global_dir / "conventional_metrics.csv",
             conv_df[conv_df["rule"].isin(["pird", "global"])])

    # ---- Adaptive attacks -------------------------------------------------
    adaptive = run_hybrid_adaptive(cfg, prepare, cache)
    sweep = adaptive["rows"]
    save_csv(hybrid_dir / "adaptive_sweep.csv", sweep)
    agg = _aggregate_adaptive(sweep)
    save_csv(hybrid_dir / "adaptive_boundary.csv", agg)

    # ---- Gradual attacks --------------------------------------------------
    if cfg.get("run_gradual", True):
        gradual = run_hybrid_gradual(cfg, prepare, cache)
        save_csv(hybrid_dir / "gradual_results.csv", gradual["results"])
        if len(gradual["trajectories"]):
            gradual["trajectories"].to_csv(hybrid_dir / "gradual_trajectories.csv", index=False)
        gagg = (
            _aggregate_gradual(gradual["results"])
            if len(gradual["results"])
            else pd.DataFrame()
        )
        save_csv(hybrid_dir / "gradual_boundary.csv", gagg)
    else:
        gradual = {"results": pd.DataFrame(), "trajectories": pd.DataFrame()}
        gagg = pd.DataFrame()
    # Global-only views.
    save_csv(global_dir / "adaptive_boundary.csv", agg[agg["rule"] == "global"])
    if len(gagg):
        save_csv(global_dir / "gradual_boundary.csv", gagg[gagg["rule"] == "global"])

    # ---- Ablation (already in sweep; extract) -----------------------------
    abl = sweep[sweep["rule"].str.startswith("global_")].copy()
    if len(abl):
        save_csv(hybrid_dir / "ablation.csv", abl)

    # ---- Summary ----------------------------------------------------------
    boundary = {}
    for rule, g in agg.groupby("rule"):
        sev = g["severity"].to_numpy(float)
        boundary[rule] = {
            "raw": summarize_boundary(sev, g["detection_probability"].to_numpy(float)),
            "significant": summarize_boundary(
                sev, g["detection_probability_significant"].to_numpy(float)
            ),
        }
    save_json(
        hybrid_dir / "summary.json",
        {
            "meta": _experiment_meta(cfg, "hybrid", seed),
            "global": cfg.get("global", {}),
            "hybrid": cfg.get("hybrid", {}),
            "attack": cfg.get("attack", {}),
            "adaptive_boundary": boundary,
            "n_adaptive_rows": int(len(sweep)),
        },
    )
    return {"conventional": conv_df, "fixed_fpr": fixed_df, "adaptive": agg,
            "gradual": gagg, "trajectories": gradual["trajectories"],
            "complementarity": pd.DataFrame(comp_rows),
            "correlation": pd.DataFrame(corr_rows),
            "out_root": out_root}


__all__ = ["run_hybrid", "run_hybrid_adaptive", "run_hybrid_gradual"]
