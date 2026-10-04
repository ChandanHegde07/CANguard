"""Phase 3 — source→target transfer runner.

Fits PIRD entirely on a **source** domain (dataset / capture / vehicle), freezes
the per-ID statistics, Isolation Forest and threshold, then evaluates on a
**target** domain. Zero-shot transfer never uses target normal data.

Modes:
  * ``dataset``  — HCRL ↔ ROAD (GEM-CAN / Survival pending, documented)
  * ``capture``  — ROAD capture A → capture B (independent captures)
  * ``vehicle``  — Vehicle A → Vehicle B (HCRL Survival; pending, no data)

Usage:
    python -m experiments.runners.run_phase3_transfer --config <yaml> --mode dataset
"""

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

from canguard.data.gem_can import GemCanLoader  # noqa: E402
from canguard.data.survival import VEHICLE_DISPLAY, SurvivalLoader  # noqa: E402
from canguard.detectors import create_detector  # noqa: E402
from canguard.evaluation.transfer import evaluate_transfer, fit_pird_source  # noqa: E402
from canguard.evaluation.transfer_metrics import (  # noqa: E402
    canonicalize_frame_ids,
)
from canguard.evaluation.transfer_shift import (  # noqa: E402
    feature_shift_table,
    per_id_smd,
    summarize_shift,
)
from canguard.exp import (  # noqa: E402
    FeatureCache,
    load_config,
    save_csv,
    save_json,
    set_global_seed,
)
from canguard.exp.matrix import build_window_table, resolve_data_path  # noqa: E402
from canguard.exp.road_protocol import (  # noqa: E402
    build_road_window_table,
    list_eval_captures,
    load_capture_frames,
    resolve_road_root,
)
from canguard.features import BEHAVIORAL_FEATURES_V1, temporal_split  # noqa: E402


def _feature_cols(cfg: dict) -> list[str]:
    return list(cfg.get("feature_cols") or BEHAVIORAL_FEATURES_V1)


def _make_detector(cfg: dict):
    d = dict(cfg.get("detector", {}))
    kind = d.pop("kind", "isolation_forest")
    return create_detector(kind, **d)


def _canonicalize(wt: pd.DataFrame) -> pd.DataFrame:
    wt = wt.copy()
    wt["can_id"] = canonicalize_frame_ids(wt["can_id"])
    return wt


# ---------------------------------------------------------------------------
# HCRL domain
# ---------------------------------------------------------------------------
def _hcrl_window_table(
    name: str, section: dict, cfg: dict, cache: FeatureCache
) -> pd.DataFrame:
    """Build HCRL windows using the **section** (source or target) config.

    ``data_dir`` and ``sample_size`` must come from the domain actually being
    built; using the source section for the target would silently change the
    amount of data and the file location.
    """
    wt = build_window_table(
        resolve_data_path(section.get("data_dir", "data"), name),
        window_size=int(cfg.get("window_size", 30)),
        sample_size=section.get("sample_size"),
        cache=cache,
    )
    return _canonicalize(wt)


def _hcrl_source(cfg: dict, cache: FeatureCache) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = cfg.get("split", {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4})
    section = cfg["source"]
    names = section["datasets"]
    cals, ifs, vals = [], [], []
    for name in names:
        wt = _hcrl_window_table(name, section, cfg, cache)
        calib, train, _ = temporal_split(
            wt, split["calib_frac"], split["train_frac"], split["test_frac"]
        )
        cals.append(calib[calib["is_attack"] == 0])
        tr = train[train["is_attack"] == 0]
        n_val = max(1, int(len(tr) * 0.2))
        ifs.append(tr.iloc[:-n_val])
        vals.append(tr.iloc[-n_val:])
    return (pd.concat(cals, ignore_index=True), pd.concat(ifs, ignore_index=True),
            pd.concat(vals, ignore_index=True))


def _hcrl_target(cfg: dict, cache: FeatureCache) -> pd.DataFrame:
    split = cfg.get("split", {"calib_frac": 0.4, "train_frac": 0.2, "test_frac": 0.4})
    section = cfg["target"]
    names = section.get("datasets") or cfg.get("source", {}).get("datasets")
    parts = []
    for name in names:
        wt = _hcrl_window_table(name, section, cfg, cache)
        _, _, test = temporal_split(
            wt, split["calib_frac"], split["train_frac"], split["test_frac"]
        )
        test = test.copy()
        test["attack_type"] = name
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# ROAD domain
# ---------------------------------------------------------------------------
def _road_root(cfg: dict, key: str = "target"):
    return resolve_road_root(cfg[key].get("data_dir", "road/road"))


def _road_capture_wt(
    name: str, meta: dict, root: Path, cfg: dict, cache: FeatureCache
) -> pd.DataFrame:
    max_frames = cfg.get("max_frames_per_capture")
    df = load_capture_frames(root, name, meta, max_frames=max_frames)
    if df.empty:
        return pd.DataFrame()
    log = root / "attacks" / f"{name}.log"
    key = {
        "stage": "phase3_road_windows",
        "capture": name,
        "window_size": int(cfg.get("window_size", 30)),
        "max_frames": max_frames,
        "n_frames": len(df),
        "mtime": log.stat().st_mtime if log.exists() else 0,
        "protocol": "pre_injection_v2_inj_priority",
    }
    wt = build_road_window_table(df, window_size=int(cfg.get("window_size", 30)),
                                 cache=cache, cache_key=key)
    if wt.empty:
        return wt
    wt = wt.sort_values("timestamp").reset_index(drop=True)
    if "elapsed" not in wt.columns:
        wt["elapsed"] = wt["timestamp"] - wt["timestamp"].iloc[0]
    return _canonicalize(wt)


def _road_captures(
    cfg: dict, key: str, cache: FeatureCache
) -> list[tuple[str, dict, pd.DataFrame]]:
    root = _road_root(cfg, key)
    cs = cfg[key].get("capture_sample", cfg.get("capture_sample", {}))
    captures = list_eval_captures(
        root,
        skip_masquerade=cs.get("skip_masquerade", True),
        skip_unlabeled=cs.get("skip_unlabeled", True),
        per_type=cs.get("per_type", 1),
    )
    out = []
    for name, meta in captures:
        wt = _road_capture_wt(name, meta, root, cfg, cache)
        if not wt.empty:
            out.append((name, meta, wt))
    return out


def _road_source(cfg: dict, cache: FeatureCache) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(cfg.get("seed", 0))
    pools = []
    for _, meta, wt in _road_captures(cfg, "source", cache):
        interval = meta.get("injection_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            continue
        pre = wt[wt["elapsed"] < float(interval[0])]
        pools.append(pre)
    if not pools:
        raise RuntimeError("No ROAD source pre-injection normals found")
    pool = pd.concat(pools, ignore_index=True)
    pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(pool)
    a, b = int(0.6 * n), int(0.8 * n)
    return pool.iloc[:a], pool.iloc[a:b], pool.iloc[b:]


def _road_target(cfg: dict, cache: FeatureCache) -> pd.DataFrame:
    parts = []
    for name, meta, wt in _road_captures(cfg, "target", cache):
        interval = meta.get("injection_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            continue
        start = float(interval[0])
        test = wt[wt["elapsed"] >= start].copy()
        test["attack_type"] = name
        parts.append(test)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Generic (frames -> windows) helper, reused by GEM-CAN and Survival
# ---------------------------------------------------------------------------
def _windows_from_frames(
    frames: pd.DataFrame, cfg: dict, cache: FeatureCache, stage: str, ident: str,
    extra: dict | None = None,
) -> pd.DataFrame:
    if frames.empty:
        return pd.DataFrame()
    key = {
        "stage": stage,
        "id": ident,
        "window_size": int(cfg.get("window_size", 30)),
        "n_frames": len(frames),
        "columns": list(frames.columns),
    }
    if extra:
        key.update(extra)
    wt = build_road_window_table(
        frames, window_size=int(cfg.get("window_size", 30)), cache=cache, cache_key=key
    )
    if wt.empty:
        return wt
    wt = wt.sort_values("timestamp").reset_index(drop=True)
    if "elapsed" not in wt.columns:
        wt["elapsed"] = wt["timestamp"] - wt["timestamp"].iloc[0]
    return _canonicalize(wt)


def _split_source_pool(pool: pd.DataFrame, seed: int) -> tuple:
    pool = pool.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    n = len(pool)
    a, b = int(0.6 * n), int(0.8 * n)
    return pool.iloc[:a], pool.iloc[a:b], pool.iloc[b:]


# ---------------------------------------------------------------------------
# GEM-CAN domain (29-bit extended IDs; two independent sessions)
# ---------------------------------------------------------------------------
def _gem_capture_wts(cfg: dict, section: dict, cache: FeatureCache):
    loader = GemCanLoader(section.get("data_dir", "GEM_CAN_Dataset_R2/GEM_CAN_Dataset_R2"))
    max_frames = cfg.get("max_frames_per_capture")
    out = []
    for cap in loader.list_captures():
        frames = loader.load_capture(cap, sample_size=max_frames)
        if frames.empty:
            continue
        mtime = loader.capture_path(cap).stat().st_mtime
        wt = _windows_from_frames(
            frames, cfg, cache, "phase3_gem_windows", cap,
            {"mtime": mtime, "sample": max_frames},
        )
        if not wt.empty:
            out.append((cap, wt))
    return out


def _gem_source(cfg: dict, cache: FeatureCache):
    pools = [wt[wt["is_attack"] == 0] for _, wt in _gem_capture_wts(cfg, cfg["source"], cache)]
    pools = [p for p in pools if not p.empty]
    if not pools:
        raise RuntimeError("No GEM-CAN source normals found")
    return _split_source_pool(pd.concat(pools, ignore_index=True), cfg.get("seed", 0))


def _gem_target(cfg: dict, cache: FeatureCache) -> pd.DataFrame:
    parts = []
    for cap, wt in _gem_capture_wts(cfg, cfg["target"], cache):
        w = wt.copy()
        if "attack_type" not in w.columns:
            w["attack_type"] = cap
        parts.append(w)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# HCRL Survival multi-vehicle domain
# ---------------------------------------------------------------------------
def _survival_capture_wts(cfg: dict, section: dict, vehicle: str, cache: FeatureCache):
    loader = SurvivalLoader(section.get("data_dir", "survival"))
    out = []
    for stem, path, _atype in loader.list_captures(vehicle):
        frames = loader.load_capture(vehicle, stem)
        if frames.empty:
            continue
        wt = _windows_from_frames(
            frames, cfg, cache, "phase3_survival_windows", f"{vehicle}:{stem}",
            {"mtime": path.stat().st_mtime},
        )
        if not wt.empty:
            wt = wt.copy()
            wt["vehicle"] = vehicle
            out.append((stem, wt))
    return out


def _survival_source_for(cfg: dict, section: dict, vehicle: str, cache: FeatureCache):
    pools = [wt[wt["is_attack"] == 0]
             for _, wt in _survival_capture_wts(cfg, section, vehicle, cache)]
    pools = [p for p in pools if not p.empty]
    if not pools:
        raise RuntimeError(f"No Survival source normals for vehicle {vehicle!r}")
    return _split_source_pool(pd.concat(pools, ignore_index=True), cfg.get("seed", 0))


def _survival_target_for(
    cfg: dict, section: dict, vehicle: str, cache: FeatureCache
) -> pd.DataFrame:
    parts = [wt for _, wt in _survival_capture_wts(cfg, section, vehicle, cache)]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["attack_type"] = out["attack_type"].where(out["is_attack"] == 1, "Normal")
    out["vehicle"] = vehicle
    return out


def _survival_source(cfg: dict, cache: FeatureCache):
    return _survival_source_for(cfg, cfg["source"], cfg["source"]["vehicle"], cache)


def _survival_target(cfg: dict, cache: FeatureCache) -> pd.DataFrame:
    return _survival_target_for(cfg, cfg["target"], cfg["target"]["vehicle"], cache)


# ---------------------------------------------------------------------------
# Domain dispatch
# ---------------------------------------------------------------------------
_GEM_NAMES = {"gem_can", "gem", "gemcan"}
_SURVIVAL_NAMES = {"survival", "hcrl_survival", "hcrl-survival"}


def _build_source(cfg: dict, cache: FeatureCache):
    ds = cfg["source"]["dataset"].lower()
    if ds == "hcrl":
        return _hcrl_source(cfg, cache)
    if ds == "road":
        return _road_source(cfg, cache)
    if ds in _GEM_NAMES:
        return _gem_source(cfg, cache)
    if ds in _SURVIVAL_NAMES:
        return _survival_source(cfg, cache)
    raise ValueError(f"Unsupported source dataset: {ds!r}")


def _build_target(cfg: dict, cache: FeatureCache):
    ds = cfg["target"]["dataset"].lower()
    if ds == "hcrl":
        return _hcrl_target(cfg, cache)
    if ds == "road":
        return _road_target(cfg, cache)
    if ds in _GEM_NAMES:
        return _gem_target(cfg, cache)
    if ds in _SURVIVAL_NAMES:
        return _survival_target(cfg, cache)
    raise ValueError(f"Unsupported target dataset: {ds!r}")


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------
def run_dataset_transfer(cfg: dict, out_root: Path) -> dict:
    set_global_seed(int(cfg.get("seed", 0)))
    out_root.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase3"))
    feature_cols = _feature_cols(cfg)
    save_json(out_root / "config.json", cfg)

    calib, if_fit, val = _build_source(cfg, cache)
    source = fit_pird_source(
        calib, if_fit, val, feature_cols, _make_detector(cfg),
        fpr_target=cfg.get("evaluation", {}).get("fpr_target", 0.01),
    )
    target = _build_target(cfg, cache)
    if target.empty:
        raise RuntimeError("Empty target domain")

    result = evaluate_transfer(
        source, target, feature_cols,
        fpr_targets=tuple(cfg.get("evaluation", {}).get(
            "fixed_fpr_targets", [0.001, 0.01, 0.05, 0.10])),
    )

    summary_row = {
        "source_dataset": cfg["source"]["dataset"],
        "target_dataset": cfg["target"]["dataset"],
        "source_capture": cfg["source"].get("capture", ""),
        "target_capture": cfg["target"].get("capture", ""),
        "source_vehicle": cfg["source"].get("vehicle", ""),
        "target_vehicle": cfg["target"].get("vehicle", ""),
        "n_source_frames": int(source.n_calib + source.n_if + source.n_val),
        "n_target_frames": result["n_target_frames"],
        "source_threshold": result["source_threshold"],
        "source_validation_fpr": result["source_validation_fpr"],
    }
    for k in [
        "n_source_ids", "n_target_ids", "n_shared_ids", "n_source_only_ids",
        "n_target_only_ids", "id_count_overlap", "frame_weighted_overlap",
        "target_unseen_id_window_fraction", "target_unseen_id_frame_fraction",
        "target_actual_fpr", "primary_precision", "primary_recall", "primary_f1",
        "primary_roc_auc", "primary_pr_auc", "n_known_id_windows", "n_unseen_id_windows",
        "fallback_precision", "fallback_recall", "fallback_f1", "fallback_actual_fpr",
    ]:
        summary_row[k] = result.get(k)
    # Canonical (unprefixed) metric aliases for the Phase 3 result schema.
    summary_row["precision"] = result.get("primary_precision")
    summary_row["recall"] = result.get("primary_recall")
    summary_row["f1"] = result.get("primary_f1")
    summary_row["roc_auc"] = result.get("primary_roc_auc")
    summary_row["pr_auc"] = result.get("primary_pr_auc")

    # Distribution shift: source normals vs target KNOWN-ID normals. When the
    # domains share no IDs (known-ID normals too few), fall back to a purely
    # feature-level SMD over all target normals; per-ID SMD is then undefined.
    target_norm = target[target["is_attack"] == 0].copy()
    target_known_norm = target_norm[target_norm["can_id"].isin(source.known_ids)]
    shift_source = pd.concat([calib, if_fit, val], ignore_index=True)
    use_known = len(target_known_norm) > 5
    shift_target = target_known_norm if use_known else target_norm
    if len(shift_target) > 5:
        sft = feature_shift_table(shift_source, shift_target, feature_cols)
        ssum = summarize_shift(sft)
        idsmd = (per_id_smd(shift_source, target_known_norm, feature_cols)
                 if use_known else pd.DataFrame())
    else:
        sft = pd.DataFrame()
        idsmd = pd.DataFrame()
        ssum = {"mean_abs_smd": float("nan"), "median_abs_smd": float("nan"),
                "max_abs_smd": float("nan")}
    summary_row.update(ssum)
    save_csv(out_root / "feature_shift.csv", sft)
    save_csv(out_root / "per_id_shift.csv", idsmd)
    save_csv(out_root / "per_attack.csv", pd.DataFrame(result.get("per_attack", [])))
    save_csv(out_root / "fixed_fpr.csv", pd.DataFrame(result.get("fixed_fpr", [])))
    save_csv(out_root / "summary.csv", pd.DataFrame([summary_row]))
    save_json(out_root / "summary.json", {
        "experiment_id": cfg.get("experiment_id", "phase3_transfer"),
        "result": {k: v for k, v in result.items() if k != "per_attack"},
        "source_model": {
            "n_calib": source.n_calib, "n_if": source.n_if, "n_val": source.n_val,
            "n_known_ids": len(source.known_ids), "threshold": source.threshold,
        },
    })

    # --- Target-specific recalibration diagnostic (NOT zero-shot) ---------
    recal_row = None
    if cfg.get("run_recalibration_diagnostic", True):
        cal_cfg = dict(cfg)
        cal_cfg["source"] = cfg["target"]
        cal_cfg["target"] = cfg["target"]
        # Avoid recursion: use the target dataset's own normals to calibrate.
        calib2, if2, val2 = _build_source_any(cal_cfg, cache)
        recal = fit_pird_source(
            calib2, if2, val2, feature_cols, _make_detector(cfg),
            fpr_target=cfg.get("evaluation", {}).get("fpr_target", 0.01),
        )
        rres = evaluate_transfer(recal, target, feature_cols)
        recal_row = {
            "source_dataset": cfg["target"]["dataset"],
            "target_dataset": cfg["target"]["dataset"],
            "experiment_id": cfg.get("experiment_id"), "diagnostic": "target_recalibration",
            "n_known_id_windows": rres["n_known_id_windows"],
            "n_unseen_id_windows": rres["n_unseen_id_windows"],
            "target_actual_fpr": rres.get("target_actual_fpr"),
            "primary_precision": rres.get("primary_precision"),
            "primary_recall": rres.get("primary_recall"),
            "primary_f1": rres.get("primary_f1"),
            "primary_roc_auc": rres.get("primary_roc_auc"),
            "primary_pr_auc": rres.get("primary_pr_auc"),
        }
        save_csv(out_root / "recalibration_summary.csv", pd.DataFrame([recal_row]))

    print(f"[phase3] {cfg['source']['dataset']} -> {cfg['target']['dataset']}: "
          f"known_windows={result['n_known_id_windows']} unseen={result['n_unseen_id_windows']} "
          f"shared_ids={result['n_shared_ids']}/{result['n_target_ids']} "
          f"primary_f1={result.get('primary_f1')} target_fpr={result.get('target_actual_fpr')}")
    return {"summary": summary_row, "result": result, "shift": sft,
            "recalibration": recal_row}


def _build_source_any(cfg: dict, cache: FeatureCache):
    """Build source normals for the target-recalibration diagnostic."""
    return _build_source(cfg, cache)


def run_capture_transfer(cfg: dict, out_root: Path) -> dict:
    """ROAD capture A → capture B transfer (independent captures)."""
    set_global_seed(int(cfg.get("seed", 0)))
    out_root.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase3"))
    feature_cols = _feature_cols(cfg)
    save_json(out_root / "config.json", cfg)
    root = _road_root(cfg, "source")
    cs = cfg.get("capture_sample", {"per_type": 1, "skip_masquerade": True, "skip_unlabeled": True})
    captures = list_eval_captures(
        root, skip_masquerade=cs.get("skip_masquerade", True),
        skip_unlabeled=cs.get("skip_unlabeled", True), per_type=cs.get("per_type", 1),
    )
    # source captures = those with injection; target = same set (A→B, A≠B)
    rows = []
    wts = {name: _road_capture_wt(name, meta, root, cfg, cache) for name, meta in captures}
    for src_name, src_meta in captures:
        src_wt = wts[src_name]
        if src_wt.empty:
            continue
        s_interval = src_meta.get("injection_interval")
        if not isinstance(s_interval, (list, tuple)) or len(s_interval) != 2:
            continue
        src_norm = src_wt[src_wt["elapsed"] < float(s_interval[0])]
        if len(src_norm) < 200:
            continue
        pool = src_norm.sample(
            frac=1.0, random_state=int(cfg.get("seed", 0))
        ).reset_index(drop=True)
        n = len(pool)
        a, b = int(0.6 * n), int(0.8 * n)
        source = fit_pird_source(
            pool.iloc[:a], pool.iloc[a:b], pool.iloc[b:], feature_cols,
            _make_detector(cfg), fpr_target=cfg.get("evaluation", {}).get("fpr_target", 0.01),
        )
        for tgt_name, tgt_meta in captures:
            if tgt_name == src_name:
                continue
            tgt_wt = wts[tgt_name]
            t_int = tgt_meta.get("injection_interval")
            if tgt_wt.empty or not isinstance(t_int, (list, tuple)) or len(t_int) != 2:
                continue
            target = tgt_wt[tgt_wt["elapsed"] >= float(t_int[0])].copy()
            target["attack_type"] = tgt_name
            if target.empty:
                continue
            res = evaluate_transfer(source, target, feature_cols)
            rows.append({
                "source_capture": src_name, "target_capture": tgt_name,
                "source_dataset": "road", "target_dataset": "road",
                "n_shared_ids": res["n_shared_ids"], "n_target_ids": res["n_target_ids"],
                "frame_weighted_overlap": res["frame_weighted_overlap"],
                "source_threshold": res["source_threshold"],
                "target_actual_fpr": res.get("target_actual_fpr"),
                "primary_f1": res.get("primary_f1"), "primary_recall": res.get("primary_recall"),
                "primary_precision": res.get("primary_precision"),
                "primary_roc_auc": res.get("primary_roc_auc"),
                "primary_pr_auc": res.get("primary_pr_auc"),
                "fallback_f1": res.get("fallback_f1"),
                "fallback_actual_fpr": res.get("fallback_actual_fpr"),
            })
    df = pd.DataFrame(rows)
    save_csv(out_root / "capture_transfer.csv", df)
    return {"capture_transfer": df}


def _result_summary_row(
    source, result: dict, cfg: dict, *, source_dataset: str, target_dataset: str,
    source_capture: str = "", target_capture: str = "",
    source_vehicle: str = "", target_vehicle: str = "",
) -> dict:
    row = {
        "source_dataset": source_dataset, "target_dataset": target_dataset,
        "source_capture": source_capture, "target_capture": target_capture,
        "source_vehicle": source_vehicle, "target_vehicle": target_vehicle,
        "n_source_frames": int(source.n_calib + source.n_if + source.n_val),
        "n_target_frames": result["n_target_frames"],
        "source_threshold": result["source_threshold"],
        "source_validation_fpr": result["source_validation_fpr"],
    }
    for k in [
        "n_source_ids", "n_target_ids", "n_shared_ids", "n_source_only_ids",
        "n_target_only_ids", "id_count_overlap", "frame_weighted_overlap",
        "target_unseen_id_window_fraction", "target_unseen_id_frame_fraction",
        "target_actual_fpr", "primary_precision", "primary_recall", "primary_f1",
        "primary_roc_auc", "primary_pr_auc", "n_known_id_windows", "n_unseen_id_windows",
        "fallback_precision", "fallback_recall", "fallback_f1", "fallback_actual_fpr",
    ]:
        row[k] = result.get(k)
    row["precision"] = result.get("primary_precision")
    row["recall"] = result.get("primary_recall")
    row["f1"] = result.get("primary_f1")
    row["roc_auc"] = result.get("primary_roc_auc")
    row["pr_auc"] = result.get("primary_pr_auc")
    return row


def _shift_for(source_normals: pd.DataFrame, target: pd.DataFrame, source, feature_cols: list[str]):
    """Feature/per-ID SMD for known-ID target normals (diagnostic only)."""
    target_norm = target[target["is_attack"] == 0]
    target_known = target_norm[target_norm["can_id"].isin(source.known_ids)]
    if len(target_known) > 5:
        sft = feature_shift_table(source_normals, target_known, feature_cols)
        return sft, per_id_smd(source_normals, target_known, feature_cols), summarize_shift(sft)
    return (pd.DataFrame(), pd.DataFrame(),
            {"mean_abs_smd": float("nan"), "median_abs_smd": float("nan"),
             "max_abs_smd": float("nan")})


def run_vehicle_transfer(cfg: dict, out_root: Path) -> dict:
    """Cross-vehicle transfer (Vehicle A → Vehicle B) on HCRL Survival.

    Source normals from vehicle A fit per-ID stats + Isolation Forest + threshold;
    vehicle B contributes nothing to fitting and is only evaluated. All ordered
    pairs among the available vehicles are run.
    """
    set_global_seed(int(cfg.get("seed", 0)))
    out_root.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase3"))
    feature_cols = _feature_cols(cfg)
    save_json(out_root / "config.json", cfg)
    section = cfg.get("dataset", {})
    loader = SurvivalLoader(section.get("data_dir", "survival"))
    available = loader.list_vehicles()
    vehicles = [str(v) for v in (section.get("vehicles") or available)]
    vehicles = [v for v in vehicles if v in available]
    fpr_target = cfg.get("evaluation", {}).get("fpr_target", 0.01)
    fpr_targets = tuple(cfg.get("evaluation", {}).get(
        "fixed_fpr_targets", [0.001, 0.01, 0.05, 0.10]))

    # Build each capture's window table exactly once.
    wts: dict[str, list[pd.DataFrame]] = {}
    for v in vehicles:
        wts[v] = [wt for _, wt in _survival_capture_wts(cfg, section, v, cache)]

    source_models: dict[str, object] = {}
    source_normals: dict[str, pd.DataFrame] = {}
    for v in vehicles:
        pools = [wt[wt["is_attack"] == 0] for wt in wts[v] if not wt.empty]
        pools = [p for p in pools if not p.empty]
        if not pools:
            source_models[v] = None
            continue
        calib, ifv, val = _split_source_pool(
            pd.concat(pools, ignore_index=True), cfg.get("seed", 0))
        if min(len(calib), len(ifv), len(val)) == 0:
            source_models[v] = None
            continue
        source_models[v] = fit_pird_source(
            calib, ifv, val, feature_cols, _make_detector(cfg), fpr_target=fpr_target)
        source_normals[v] = pd.concat([calib, ifv, val], ignore_index=True)

    rows: list[dict] = []
    attack_frames: list[pd.DataFrame] = []
    for src_v in vehicles:
        source = source_models.get(src_v)
        if source is None:
            print(f"[phase3] vehicle {src_v}: insufficient source normals; skipped as source")
            continue
        for tgt_v in vehicles:
            if tgt_v == src_v:
                continue
            target = pd.concat([wt for wt in wts[tgt_v] if not wt.empty], ignore_index=True)
            if target.empty:
                continue
            res = evaluate_transfer(source, target, feature_cols, fpr_targets=fpr_targets)
            pair_dir = out_root / f"{src_v}_to_{tgt_v}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            row = _result_summary_row(
                source, res, cfg, source_dataset="survival", target_dataset="survival",
                source_vehicle=src_v, target_vehicle=tgt_v)
            sft, idsmd, ssum = _shift_for(source_normals[src_v], target, source, feature_cols)
            row.update(ssum)
            save_csv(pair_dir / "feature_shift.csv", sft)
            save_csv(pair_dir / "per_id_shift.csv", idsmd)
            save_csv(pair_dir / "summary.csv", pd.DataFrame([row]))
            save_csv(pair_dir / "fixed_fpr.csv", pd.DataFrame(res.get("fixed_fpr", [])))
            pa = pd.DataFrame(res.get("per_attack", []))
            if not pa.empty:
                pa.insert(0, "target_vehicle", tgt_v)
                pa.insert(0, "source_vehicle", src_v)
                attack_frames.append(pa)
                save_csv(pair_dir / "per_attack.csv", pa)
            rows.append(row)
            print(f"[phase3] {src_v} -> {tgt_v}: shared={res['n_shared_ids']}/"
                  f"{res['n_target_ids']} recall={res.get('primary_recall')} "
                  f"f1={res.get('primary_f1')} fpr={res.get('target_actual_fpr')}")

    # Target-specific recalibration diagnostic (NOT zero-shot).
    recal_rows = []
    if cfg.get("run_recalibration_diagnostic", True):
        for tgt_v in vehicles:
            pools = [wt[wt["is_attack"] == 0] for wt in wts[tgt_v] if not wt.empty]
            pools = [p for p in pools if not p.empty]
            if not pools:
                continue
            calib, ifv, val = _split_source_pool(
                pd.concat(pools, ignore_index=True), cfg.get("seed", 0))
            if min(len(calib), len(ifv), len(val)) == 0:
                continue
            recal = fit_pird_source(calib, ifv, val, feature_cols, _make_detector(cfg),
                                    fpr_target=fpr_target)
            target = pd.concat([wt for wt in wts[tgt_v] if not wt.empty], ignore_index=True)
            rres = evaluate_transfer(recal, target, feature_cols)
            recal_rows.append({
                "source_vehicle": tgt_v, "target_vehicle": tgt_v,
                "diagnostic": "target_recalibration",
                "target_actual_fpr": rres.get("target_actual_fpr"),
                "primary_precision": rres.get("primary_precision"),
                "primary_recall": rres.get("primary_recall"),
                "primary_f1": rres.get("primary_f1"),
                "primary_roc_auc": rres.get("primary_roc_auc"),
                "primary_pr_auc": rres.get("primary_pr_auc"),
            })

    summary_df = pd.DataFrame(rows)
    save_csv(out_root / "vehicle_transfer.csv", summary_df)
    save_csv(out_root / "per_attack.csv",
             pd.concat(attack_frames, ignore_index=True) if attack_frames else pd.DataFrame())
    save_csv(out_root / "recalibration_summary.csv", pd.DataFrame(recal_rows))
    save_json(out_root / "summary.json", {
        "experiment_id": cfg.get("experiment_id", "phase3_vehicle_transfer"),
        "vehicles": vehicles,
        "display_names": {v: VEHICLE_DISPLAY.get(v, v) for v in vehicles},
        "n_pairs": int(len(summary_df)),
        "results": rows,
    })
    print(f"Phase 3 vehicle transfer: {len(summary_df)} pairs over {vehicles}")
    return {"vehicle_transfer": summary_df}


def run_gem_capture_transfer(cfg: dict, out_root: Path) -> dict:
    """GEM-CAN capture transfer between the two independent sessions.

    Source = the source capture's normals (fit/freeze); target = the target
    capture's full window table (evaluation only).
    """
    set_global_seed(int(cfg.get("seed", 0)))
    out_root.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(cfg.get("cache_dir", ".cache/phase3"))
    feature_cols = _feature_cols(cfg)
    save_json(out_root / "config.json", cfg)
    section = cfg.get("dataset", cfg.get("source", {}))
    captures = _gem_capture_wts(cfg, section, cache)
    fpr_target = cfg.get("evaluation", {}).get("fpr_target", 0.01)
    fpr_targets = tuple(cfg.get("evaluation", {}).get(
        "fixed_fpr_targets", [0.001, 0.01, 0.05, 0.10]))
    rows: list[dict] = []
    for src_cap, src_wt in captures:
        norm = src_wt[src_wt["is_attack"] == 0]
        if len(norm) < 50:
            continue
        calib, ifv, val = _split_source_pool(norm, cfg.get("seed", 0))
        if min(len(calib), len(ifv), len(val)) == 0:
            continue
        source = fit_pird_source(calib, ifv, val, feature_cols, _make_detector(cfg),
                                 fpr_target=fpr_target)
        for tgt_cap, tgt_wt in captures:
            if tgt_cap == src_cap or tgt_wt.empty:
                continue
            target = tgt_wt.copy()
            if "attack_type" not in target.columns:
                target["attack_type"] = tgt_cap
            res = evaluate_transfer(source, target, feature_cols, fpr_targets=fpr_targets)
            row = _result_summary_row(
                source, res, cfg, source_dataset="gem_can", target_dataset="gem_can",
                source_capture=src_cap, target_capture=tgt_cap)
            rows.append(row)
            print(f"[phase3] gem {src_cap} -> {tgt_cap}: recall={res.get('primary_recall')} "
                  f"f1={res.get('primary_f1')} fpr={res.get('target_actual_fpr')}")
    df = pd.DataFrame(rows)
    save_csv(out_root / "capture_transfer.csv", df)
    return {"capture_transfer": df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 transfer runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode",
                        choices=["dataset", "capture", "gem_capture", "vehicle"],
                        default="dataset")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.get("output_dir", "experiments/phase3/transfer"))
    if args.mode == "dataset":
        run_dataset_transfer(cfg, out)
    elif args.mode == "capture":
        run_capture_transfer(cfg, out)
    elif args.mode == "gem_capture":
        run_gem_capture_transfer(cfg, out)
    else:
        run_vehicle_transfer(cfg, out)
    print(f"Phase 3 {args.mode} transfer complete. Results under {out}")


if __name__ == "__main__":
    main()
