"""Phase 4B — representation-density vs next-event scoring on frozen models.

Experiment A: train one frozen model per (source, backbone) with the Phase 4
protocol, then score the *same representations* with several anomaly scores and
compare zero-shot transfer. All density statistics are source-only; inference
representations are causal (online protocol).

Modes:
  score_compare : the main table (representation x score x target) + diagnostics.
  shift         : representation-shift diagnostics (same code path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.evaluation.zero_shot import (  # noqa: E402
    aggregate_frame_scores,
    evaluate_zero_shot_scores,
)
from canguard.exp import load_config, save_csv, save_json, set_global_seed  # noqa: E402
from canguard.foundation.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from canguard.foundation.data import (  # noqa: E402
    build_windows_per_capture,
    load_domain_frames,
)
from canguard.foundation.density import (  # noqa: E402
    apply_standardizer,
    distance_ratio,
    fit_centroid,
    fit_knn_reference,
    fit_mahalanobis,
    fit_standardizer,
    linear_probe,
    representation_shift,
    score_centroid,
    score_mahalanobis,
)
from canguard.foundation.pipeline import (  # noqa: E402
    SourceModel,
    _select_normal_windows,
    load_domain_windows,
    split_captures_source_val,
    train_source,
)
from canguard.foundation.scoring import (  # noqa: E402
    embed_positions,
    fit_residual_weights,
    score_next_event,
)
from canguard.foundation.tokenizer import TokenizerConfig  # noqa: E402
from canguard.foundation.train import TrainConfig  # noqa: E402

SCORE_METHODS = ["next_event", "mahalanobis", "mahalanobis_diag", "centroid", "knn"]
K_VALUES = [1, 5, 10, 20]


def _spec(cfg: dict) -> TokenizerConfig:
    tk = dict(cfg.get("tokenizer", {}))
    return TokenizerConfig(
        payload_mode=tk.get("payload_mode", "byte"),
        id_mode=tk.get("id_mode", "none"),
        id_bits=int(tk.get("id_bits", 11)),
    )


def _model_cfg(cfg: dict, backbone: str) -> dict:
    m = dict(cfg.get("model", {}))
    m["backbone"] = backbone
    return {"model": m}


def _train_cfg(cfg: dict) -> TrainConfig:
    tr = dict(cfg.get("train", {}))
    tr.setdefault("seed", int(cfg.get("seed", 0)))
    return TrainConfig(**tr)


def _device_of(model):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def _map_to_frames(Z, frame_row, n_frames):
    d = Z.shape[-1]
    out = np.full((n_frames, d), np.nan, dtype=np.float64)
    out[frame_row.reshape(-1)] = Z.reshape(-1, d)
    return out


def _stratified_idx(capture, attack, cap, seed):
    rng = np.random.default_rng(seed)
    idx = []
    keys = {(c, a) for c, a in zip(capture, attack)}
    per = max(1, cap // max(1, len(keys)))
    for c, a in keys:
        pool = np.flatnonzero((capture == c) & (attack == a))
        if len(pool) > per:
            pool = rng.choice(pool, per, replace=False)
        idx.append(pool)
    return np.sort(np.concatenate(idx)) if idx else np.zeros(0, int)


def _knn_all_k(Z, refs, kmax=20):
    from sklearn.neighbors import NearestNeighbors

    k = int(min(kmax, len(refs)))
    if k < 1:
        return {}
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(refs)
    dist, _ = nn.kneighbors(Z)
    return {kk: dist[:, :kk].mean(axis=1) for kk in K_VALUES if kk <= k}


def _val_windows(source_domains, spec, length, cfg):
    Xs, Vs, val_caps_by_domain = [], [], {}
    for domain in source_domains:
        frames, windows = load_domain_windows(
            domain,
            spec,
            length,
            length,
            roots=cfg.get("roots"),
            max_frames_per_capture=cfg.get("max_frames_per_capture", 120_000),
            cache_dir=cfg.get("cache_dir", ".cache/phase4"),
        )
        if len(windows["X"]) == 0:
            continue
        caps = sorted(set(windows["capture"]))
        _, val_caps = split_captures_source_val(caps)
        val_caps_by_domain[domain] = val_caps
        vw = _select_normal_windows(windows, frames, val_caps)
        cap = int(cfg.get("max_val_windows", 4000))
        if len(vw["X"]) > cap:
            idx = np.random.default_rng(cfg.get("seed", 0)).choice(len(vw["X"]), cap, replace=False)
            m = len(vw["X"])
            vw = {
                k: (v[idx] if isinstance(v, np.ndarray) and v.shape[0] == m else v)
                for k, v in vw.items()
            }
        Xs.append(vw["X"])
        Vs.append(vw["valid"])
    X = np.concatenate(Xs) if Xs else np.zeros((0, length, spec.feature_dim), np.float32)
    V = np.concatenate(Vs) if Vs else np.zeros((0, length, 8), bool)
    return X, V, val_caps_by_domain


def _fit_scorers(model, X_val, device, cfg):
    cap = cfg.get("max_val_repr", 40000)
    Xv = X_val
    if len(Xv) > cap:
        Xv = Xv[np.random.default_rng(cfg.get("seed", 0)).choice(len(Xv), cap, replace=False)]
    Z = embed_positions(model, Xv, device=device, causal=True).reshape(-1, model.d_model)
    Z = Z.astype(np.float64)
    mu, sd = fit_standardizer(Z)
    Zv = apply_standardizer(Z, mu, sd)
    return {
        "standardizer": (mu, sd),
        "mahalanobis": fit_mahalanobis(Zv, shrinkage=float(cfg.get("shrinkage", 0.1))),
        "mahalanobis_diag": fit_mahalanobis(Zv, diagonal=True),
        "centroid": fit_centroid(Zv),
        "knn_refs": fit_knn_reference(
            Zv, max_refs=int(cfg.get("knn_refs", 3000)), seed=cfg.get("seed", 0)
        ),
        "val_standardized": Zv,
    }


def _val_scores(scorers):
    Zv = scorers["val_standardized"]
    out = {
        "mahalanobis": score_mahalanobis(Zv, scorers["mahalanobis"]),
        "mahalanobis_diag": score_mahalanobis(Zv, scorers["mahalanobis_diag"]),
        "centroid": score_centroid(Zv, scorers["centroid"]),
    }
    out["knn"] = _knn_all_k(Zv, scorers["knn_refs"])
    return out


def _score_target(model, frames, spec, length, weights, device):
    windows = build_windows_per_capture(frames, spec, length, length)
    n = len(frames)
    out = {
        "labels": frames["is_attack"].to_numpy(int),
        "capture": frames["capture"].to_numpy(),
        "times": frames["timestamp"].to_numpy(float),
    }
    if len(windows["X"]) == 0:
        out["next_event"] = np.full(n, np.nan)
        out["_Zf"] = np.full((n, model.d_model), np.nan)
        return out
    ne = score_next_event(model, windows["X"], windows["valid"], spec, weights, device=device)
    out["next_event"] = aggregate_frame_scores(ne, windows["frame_row"], n, position_offset=1)[0]
    Z = embed_positions(model, windows["X"], device=device, causal=True)
    out["_Zf"] = _map_to_frames(Z, windows["frame_row"], n)
    return out


def _source_probe_data(source, spec, length, model, device, mu, sd, cfg):
    Zs, ys = [], []
    cap = int(cfg.get("probe_frames_per_domain", 20000))
    for domain in source:
        frames = load_domain_frames(
            domain,
            roots=cfg.get("roots"),
            max_frames_per_capture=cfg.get("max_frames_per_capture", 120_000),
            cache=None,
        )
        if frames.empty:
            continue
        windows = build_windows_per_capture(frames, spec, length, length)
        if len(windows["X"]) == 0:
            continue
        Zf = _map_to_frames(
            embed_positions(model, windows["X"], device=device, causal=True),
            windows["frame_row"],
            len(frames),
        )
        lab = frames["is_attack"].to_numpy(int)
        ok = np.isfinite(Zf).all(axis=1)
        Zs.append(Zf[ok])
        ys.append(lab[ok])
    if not Zs:
        return np.zeros((0, model.d_model)), np.zeros(0, int)
    Z = apply_standardizer(np.concatenate(Zs), mu, sd)
    y = np.concatenate(ys)
    rng = np.random.default_rng(cfg.get("seed", 0))
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    k = min(cap, len(pos), len(neg))
    if k == 0:
        return np.zeros((0, model.d_model)), np.zeros(0, int)
    sel = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    return Z[sel], y[sel]


def _source_ids(source, cfg):
    ids = set()
    for domain in source:
        frames = load_domain_frames(
            domain,
            roots=cfg.get("roots"),
            max_frames_per_capture=cfg.get("max_frames_per_capture", 120_000),
            cache=None,
        )
        if not frames.empty:
            ids.update(frames["can_id"].astype(str).unique().tolist())
    return ids


def _fit_or_load(cfg, source, skey, backbone, spec, length, ckpt: Path) -> SourceModel:
    if cfg.get("reuse_checkpoints", True) and ckpt.exists():
        model, sp, _meta = load_checkpoint(ckpt)
        device = _device_of(model)
        X_val, V_val, _ = _val_windows(source, spec, length, cfg)
        weights = fit_residual_weights(model, X_val, V_val, spec, device)
        vs = score_next_event(model, X_val, V_val, spec, weights, device).reshape(-1)
        vs = vs[np.isfinite(vs)]
        thr = (
            float(np.percentile(vs, (1 - float(cfg.get("fpr_target", 0.01))) * 100))
            if vs.size
            else float("inf")
        )
        print(f"[phase4b] loaded checkpoint {ckpt.name}", flush=True)
        return SourceModel(
            model=model,
            spec=sp,
            weights=weights,
            threshold=thr,
            source_val_normal_scores=vs,
            train_history={},
            n_train_windows=0,
            n_val_windows=0,
            source_val_fpr=float((vs >= thr).mean()) if vs.size else float("nan"),
            source_ids=_source_ids(source, cfg),
        )
    src = train_source(
        source,
        spec,
        _model_cfg(cfg, backbone),
        _train_cfg(cfg),
        length,
        roots=cfg.get("roots"),
        max_frames_per_capture=cfg.get("max_frames_per_capture", 120_000),
        fpr_target=float(cfg.get("fpr_target", 0.01)),
        max_train_windows=cfg.get("max_train_windows", 30000),
        cache_dir=cfg.get("cache_dir", ".cache/phase4"),
    )
    save_checkpoint(ckpt, src.model, spec, {"source": source, "backbone": backbone})
    return src


def run_score_compare(cfg: dict, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(cfg.get("seed", 0)))
    spec = _spec(cfg)
    length = int(cfg.get("length", 128))
    seed = int(cfg.get("seed", 0))
    sources = cfg.get("sources", [["hcrl"]])
    targets = cfg.get("targets", [])
    backbones = cfg.get("backbones", ["transformer"])
    max_eval = int(cfg.get("max_eval_frames", 40000))
    run_probe = bool(cfg.get("run_probe", False))
    rows, shift_rows = [], []
    ckpt_dir = out / "checkpoints"

    for source in sources:
        skey = "+".join(source)
        for backbone in backbones:
            ckpt = ckpt_dir / f"{skey.replace('+', '_')}_{backbone}.pt"
            print(f"[phase4b] source={skey} backbone={backbone}", flush=True)
            src = _fit_or_load(cfg, source, skey, backbone, spec, length, ckpt)
            device = _device_of(src.model)
            X_val, V_val, val_caps = _val_windows(source, spec, length, cfg)
            scorers = _fit_scorers(src.model, X_val, device, cfg)
            mu, sd = scorers["standardizer"]
            v_scores = {"next_event": src.source_val_normal_scores}
            v_scores.update(_val_scores(scorers))
            if run_probe:
                probe_Z, probe_y = _source_probe_data(
                    source, spec, length, src.model, device, mu, sd, cfg
                )

            targets_for_run = list(targets)
            for domain in source:
                if val_caps.get(domain):
                    targets_for_run.append((domain, val_caps[domain]))

            for tspec in targets_for_run:
                if isinstance(tspec, tuple):
                    domain, caps = tspec
                    frames = load_domain_frames(
                        domain,
                        roots=cfg.get("roots"),
                        max_frames_per_capture=cfg.get("max_frames_per_capture", 120_000),
                        captures=caps,
                        tail_hcrl=(domain != "hcrl"),
                        cache=None,
                    )
                    target_name, subset = f"{domain}:val", "in_domain"
                else:
                    domain = tspec
                    if domain in source:
                        continue
                    frames = load_domain_frames(
                        domain,
                        roots=cfg.get("roots"),
                        max_frames_per_capture=cfg.get("max_frames_per_capture", 120_000),
                        tail_hcrl=(domain != "hcrl"),
                        cache=None,
                    )
                    target_name, subset = domain, "zero_shot"
                if frames.empty:
                    continue
                scored = _score_target(src.model, frames, spec, length, src.weights, device)
                idx = _stratified_idx(scored["capture"], scored["labels"], max_eval, seed)
                Zf = scored["_Zf"]
                Zt = Zf[idx]
                good = np.isfinite(Zt).all(axis=1)
                Zt_std = np.full_like(Zt, np.nan)
                Zt_std[good] = apply_standardizer(Zt[good], mu, sd)
                method_scores = {"next_event": scored["next_event"][idx]}
                method_scores["mahalanobis"] = _nan_scores(
                    score_mahalanobis, Zt_std, scorers["mahalanobis"]
                )
                method_scores["mahalanobis_diag"] = _nan_scores(
                    score_mahalanobis, Zt_std, scorers["mahalanobis_diag"]
                )
                method_scores["centroid"] = _nan_scores(score_centroid, Zt_std, scorers["centroid"])
                method_scores["knn"] = _knn_target(Zt_std, scorers["knn_refs"])

                Zt_norm = Zf[(scored["labels"] == 0) & np.isfinite(Zf).all(axis=1)]
                Zt_atk = Zf[(scored["labels"] == 1) & np.isfinite(Zf).all(axis=1)]
                try:
                    sh = representation_shift(scorers["val_standardized"], Zt_norm, seed=seed)
                    se = distance_ratio(scorers["val_standardized"], Zt_norm, Zt_atk)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"    [warn] diagnostics failed for {target_name}: {exc}", flush=True)
                    sh = {
                        "centroid_distance": float("nan"),
                        "mean_abs_smd": float("nan"),
                        "median_abs_smd": float("nan"),
                        "max_abs_smd": float("nan"),
                        "mmd_rbf": float("nan"),
                        "source_centroid_norm": float("nan"),
                    }
                    se = {
                        "intra_normal_distance": float("nan"),
                        "normal_to_source_centroid": float("nan"),
                        "attack_to_target_normal_centroid": float("nan"),
                        "attack_normal_distance_ratio": float("nan"),
                    }
                shift_rows.append(
                    {
                        "source": skey,
                        "target": target_name,
                        "subset": subset,
                        "model": backbone,
                        **sh,
                        **se,
                    }
                )

                base = {
                    "source": skey,
                    "target": target_name,
                    "subset": subset,
                    "model": backbone,
                    "representation": "id_free_causal",
                    "objective": cfg.get("train", {}).get("objective", "next_event"),
                    "context_length": length,
                    "seed": seed,
                    "pretraining_diversity": len(source),
                    "normal_shift": sh["mean_abs_smd"],
                    "attack_separation": se["attack_normal_distance_ratio"],
                }
                method_rows = []
                for method in SCORE_METHODS:
                    ms = method_scores[method]
                    if isinstance(ms, dict):
                        for kk, sv in ms.items():
                            res = evaluate_zero_shot_scores(
                                v_scores["knn"][kk], sv, scored["labels"][idx]
                            )
                            r = _metrics_row(base, res)
                            r.update({"score": "knn", "k": kk})
                            method_rows.append(r)
                    else:
                        res = evaluate_zero_shot_scores(v_scores[method], ms, scored["labels"][idx])
                        r = _metrics_row(base, res)
                        r.update({"score": method, "k": np.nan})
                        method_rows.append(r)
                if run_probe and len(probe_y):
                    pr = linear_probe(probe_Z, probe_y, Zt[good], scored["labels"][idx][good])
                    for r in method_rows:
                        r.update(pr)
                rows.extend(method_rows)
                best = max(
                    method_rows,
                    key=lambda r: (r["tpr_at_1pct"] if np.isfinite(r["tpr_at_1pct"]) else -1),
                )
                print(
                    f"    {skey}->{target_name} [{subset}] ne_f1={method_rows[0]['f1']:.3f} "
                    f"best={best['score']} tpr1={best['tpr_at_1pct']:.3f} "
                    f"shift={sh['mean_abs_smd']:.3f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    save_csv(out / "score_comparison.csv", df)
    save_csv(out / "representation_shift.csv", pd.DataFrame(shift_rows))
    save_json(
        out / "phase4b_summary.json",
        {
            "n_rows": int(len(df)),
            "sources": ["+".join(s) for s in sources],
            "backbones": backbones,
            "context_length": length,
            "seed": seed,
        },
    )
    return df


def _nan_scores(fn, Z, arg):
    out = np.full(len(Z), np.nan)
    good = np.isfinite(Z).all(axis=1)
    if good.any():
        out[good] = fn(Z[good], arg)
    return out


def _knn_target(Z, refs, kmax=20):
    good = np.isfinite(Z).all(axis=1)
    out = {kk: np.full(len(Z), np.nan) for kk in K_VALUES}
    if good.any():
        for kk, vals in _knn_all_k(Z[good], refs, kmax=kmax).items():
            out[kk][good] = vals
    return out


def _metrics_row(base, res):
    row = dict(base)
    row.update(
        {
            "recall": res.get("recall"),
            "precision": res.get("precision"),
            "f1": res.get("f1"),
            "roc_auc": res.get("roc_auc"),
            "pr_auc": res.get("pr_auc"),
            "actual_fpr": res.get("target_actual_fpr"),
            "source_validation_fpr": res.get("source_validation_fpr"),
            "source_threshold": res.get("source_threshold"),
            "tpr_at_1pct": res.get("tpr_at_0.01_fpr"),
            "tpr_at_5pct": res.get("tpr_at_0.05_fpr"),
            "n_target": res.get("n_target"),
            "n_target_normal": res.get("n_target_normal"),
            "n_target_attack": res.get("n_target_attack"),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4B runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["score_compare", "shift"], default="score_compare")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.get("output_dir", "experiments/phase4b/results/score_compare"))
    run_score_compare(cfg, out)
    print(f"Phase 4B {args.mode} complete. Results under {out}")


if __name__ == "__main__":
    main()
