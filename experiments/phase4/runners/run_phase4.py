"""Phase 4 — CAN-TFM experiment runner.

Modes
-----
manifest    : dataset provenance manifest (normal/attack frame counts).
prototype   : in-domain HCRL end-to-end for each backbone (milestone 1).
zero_shot   : frozen source model(s) -> target domain(s); no target fitting.
ablation    : id representation / context length / backbone / objective / diversity.

Every mode writes machine-readable results with the Phase 4 schema. Target data is
never passed to `train_source`; the target-recalibration diagnostic (if enabled)
is a separate, explicitly labelled output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from canguard.evaluation.metrics import compute_metrics  # noqa: E402
from canguard.exp import load_config, save_csv, save_json, set_global_seed  # noqa: E402
from canguard.exp.cache import FeatureCache  # noqa: E402
from canguard.foundation.data import (  # noqa: E402
    all_domains,
    dataset_manifest,
    load_domain_frames,
)
from canguard.foundation.pipeline import (  # noqa: E402
    evaluate_source_on_target,
    train_source,
    unseen_id_subset,
)
from canguard.foundation.tokenizer import TokenizerConfig  # noqa: E402
from canguard.foundation.train import TrainConfig  # noqa: E402

RESULT_COLS = [
    "experiment",
    "source",
    "target",
    "model",
    "id_mode",
    "payload_mode",
    "length",
    "objective",
    "backbone",
    "attack",
    "subset",
    "n_source_ids",
    "n_target_ids",
    "n_shared_ids",
    "n_target",
    "n_target_normal",
    "n_target_attack",
    "recall",
    "precision",
    "f1",
    "roc_auc",
    "pr_auc",
    "source_threshold",
    "source_validation_fpr",
    "actual_fpr",
    "tpr_at_1pct_fpr",
    "tpr_at_5pct_fpr",
    "detection_delay_s",
    "detection_rate",
    "delay_n_episodes",
]


def _spec(cfg: dict) -> TokenizerConfig:
    tk = dict(cfg.get("tokenizer", {}))
    return TokenizerConfig(
        payload_mode=tk.get("payload_mode", "byte"),
        id_mode=tk.get("id_mode", "none"),
        id_bits=int(tk.get("id_bits", 11)),
    )


def _model_cfg(cfg: dict) -> dict:
    return {"model": cfg.get("model", {})}


def _train_cfg(cfg: dict) -> TrainConfig:
    tr = dict(cfg.get("train", {}))
    tr.setdefault("seed", int(cfg.get("seed", 0)))
    return TrainConfig(**tr)


def _base_row(cfg: dict, source: list[str], target: str, backbone: str, objective: str) -> dict:
    return {
        "experiment": cfg.get("experiment_id", "phase4"),
        "source": "+".join(source),
        "target": target,
        "model": "can_tfm",
        "id_mode": cfg.get("tokenizer", {}).get("id_mode", "none"),
        "payload_mode": cfg.get("tokenizer", {}).get("payload_mode", "byte"),
        "length": cfg.get("length", 128),
        "objective": objective,
        "backbone": backbone,
        "attack": "ALL",
    }


def _per_attack(scored: dict, threshold: float) -> list[dict]:
    df = pd.DataFrame(
        {
            "scores": scored["scores"],
            "labels": scored["labels"],
            "attack_type": scored["attack_type"],
        }
    ).dropna(subset=["scores"])
    rows = []
    for atype, g in df.groupby("attack_type"):
        if atype == "Normal":
            continue
        y = g["labels"].to_numpy(int)
        pred = (g["scores"].to_numpy(float) >= threshold).astype(int)
        m = compute_metrics(y, pred, g["scores"].to_numpy(float))
        rows.append(
            {
                "attack": str(atype),
                "n_target_attack": int(y.sum()),
                "recall": m["recall"],
                "precision": m["precision"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
            }
        )
    return rows


def _row_from_result(base: dict, res: dict, cfg: dict) -> dict:
    row = dict(base)
    row.update(
        {
            "n_target": res.get("n_target"),
            "n_target_normal": res.get("n_target_normal"),
            "n_target_attack": res.get("n_target_attack"),
            "recall": res.get("recall"),
            "precision": res.get("precision"),
            "f1": res.get("f1"),
            "roc_auc": res.get("roc_auc"),
            "pr_auc": res.get("pr_auc"),
            "source_threshold": res.get("source_threshold"),
            "source_validation_fpr": res.get("source_validation_fpr"),
            "actual_fpr": res.get("target_actual_fpr"),
            "tpr_at_1pct_fpr": res.get("tpr_at_0.01_fpr"),
            "tpr_at_5pct_fpr": res.get("tpr_at_0.05_fpr"),
            "detection_delay_s": res.get("delay_median_delay_s"),
            "detection_rate": res.get("delay_detection_rate"),
            "delay_n_episodes": res.get("delay_n_episodes"),
        }
    )
    return row


def _source_key(
    source: list[str], spec: TokenizerConfig, cfg: dict, backbone: str, objective: str
) -> str:
    payload = {
        "source": sorted(source),
        "spec": spec.to_dict(),
        "cfg": cfg.get("model", {}),
        "train": cfg.get("train", {}),
        "length": cfg.get("length"),
        "max_frames": cfg.get("max_frames_per_capture"),
        "backbone_override": backbone,
        "objective": objective,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def _train(
    cfg: dict, source: list[str], spec: TokenizerConfig, backbone: str, objective: str, cache: dict
) -> object:
    key = _source_key(source, spec, cfg, backbone, objective)
    if key in cache:
        return cache[key]
    model_cfg = _model_cfg(cfg)
    model_cfg["model"] = dict(model_cfg["model"], backbone=backbone)
    tr = _train_cfg(cfg)
    tr.objective = objective
    src = train_source(
        source,
        spec,
        model_cfg,
        tr,
        int(cfg.get("length", 128)),
        roots=cfg.get("roots"),
        max_frames_per_capture=cfg.get("max_frames_per_capture", 150_000),
        fpr_target=float(cfg.get("fpr_target", 0.01)),
        max_train_windows=cfg.get("max_train_windows", 20_000),
        cache_dir=cfg.get("cache_dir", ".cache/phase4"),
    )
    cache[key] = src
    return src


_FRAME_CACHE = FeatureCache(".cache/phase4")


def _target_frames(cfg: dict, target: str) -> pd.DataFrame:
    # HCRL attacks live in the early/middle region; the tail is clean normal and
    # is reserved for source pretraining. For evaluation we therefore take the
    # head slice for HCRL (tail_hcrl=False) and the default tail elsewhere.
    return load_domain_frames(
        target,
        roots=cfg.get("roots"),
        max_frames_per_capture=cfg.get("max_frames_per_capture", 150_000),
        captures=cfg.get("target_captures", {}).get(target),
        tail_hcrl=(target != "hcrl"),
        cache=_FRAME_CACHE,
    )


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_manifest(cfg: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    domains = cfg.get("domains", all_domains())
    records = dataset_manifest(
        domains,
        roots=cfg.get("roots"),
        max_frames_per_capture=cfg.get("max_frames_per_capture", 150_000),
    )
    save_csv(out / "dataset_manifest.csv", pd.DataFrame(records))
    save_json(out / "dataset_manifest.json", records)
    print(f"[phase4] manifest: {len(records)} captures across {len(domains)} domains")


def run_prototype(cfg: dict, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(cfg.get("seed", 0)))
    spec = _spec(cfg)
    source = cfg.get("source", ["hcrl"])
    target = (
        cfg.get("target", source[0])
        if isinstance(cfg.get("target"), str)
        else cfg.get("target", source)[0]
    )
    length = int(cfg.get("length", 128))
    rows = []
    for backbone in cfg.get("backbones", ["mlp", "gru", "transformer"]):
        src = _train(
            cfg, source, spec, backbone, cfg.get("train", {}).get("objective", "next_event"), {}
        )
        frames = _target_frames(cfg, target)
        res = evaluate_source_on_target(
            src, frames, length, fpr_target=float(cfg.get("fpr_target", 0.01))
        )
        base = _base_row(
            cfg, source, target, backbone, cfg.get("train", {}).get("objective", "next_event")
        )
        rows.append(_row_from_result(base, res, cfg))
        print(
            f"[phase4] prototype {backbone}: f1={res['f1']:.3f} fpr={res['target_actual_fpr']:.4f} "
            f"tpr@1%={res.get('tpr_at_0.01_fpr')}"
        )
    df = pd.DataFrame(rows)
    save_csv(out / "prototype_results.csv", df)
    return df


def run_zero_shot(cfg: dict, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(cfg.get("seed", 0)))
    spec = _spec(cfg)
    length = int(cfg.get("length", 128))
    backbone = cfg.get("model", {}).get("backbone", "transformer")
    objective = cfg.get("train", {}).get("objective", "next_event")
    sources = cfg.get("sources", [["hcrl"]])
    targets = cfg.get("targets", [d for d in all_domains()])
    cache: dict = {}
    rows, recal_rows = [], []
    for source in sources:
        src = _train(cfg, source, spec, backbone, objective, cache)
        target_cache: dict[str, pd.DataFrame] = {}
        for target in targets:
            if target in source:
                continue
            if target not in target_cache:
                target_cache[target] = _target_frames(cfg, target)
            frames = target_cache[target]
            if frames.empty:
                continue
            res = evaluate_source_on_target(
                src, frames, length, fpr_target=float(cfg.get("fpr_target", 0.01))
            )
            base = _base_row(cfg, source, target, backbone, objective)
            rows.append(_row_from_result(base, res, cfg))
            print(
                f"[phase4] {('+'.join(source))}->{target}: f1={res['f1']:.3f} "
                f"fpr={res['target_actual_fpr']:.4f} tpr@1%={res.get('tpr_at_0.01_fpr')}"
            )
        # target-recalibration diagnostic (explicitly NOT zero-shot)
        if cfg.get("run_recalibration_diagnostic", False):
            for target in targets:
                if target in source:
                    continue
                try:
                    recal = _train(cfg, [target], spec, backbone, objective, cache)
                    frames = (
                        target_cache.get(target)
                        if target in target_cache
                        else _target_frames(cfg, target)
                    )
                    rres = evaluate_source_on_target(recal, frames, length)
                    recal_rows.append(
                        {
                            "source": target,
                            "target": target,
                            "diagnostic": "target_recalibration",
                            "f1": rres["f1"],
                            "recall": rres["recall"],
                            "actual_fpr": rres["target_actual_fpr"],
                        }
                    )
                except Exception as exc:  # pragma: no cover
                    recal_rows.append({"source": target, "target": target, "error": str(exc)})
    df = pd.DataFrame(rows)
    save_csv(out / "zero_shot_results.csv", df)
    if recal_rows:
        save_csv(out / "recalibration_diagnostic.csv", pd.DataFrame(recal_rows))
    return df


def run_cross_id(cfg: dict, out: Path) -> pd.DataFrame:
    """The key architectural test: evaluate only on target IDs unseen in source."""
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(cfg.get("seed", 0)))
    spec = _spec(cfg)
    length = int(cfg.get("length", 128))
    backbone = cfg.get("model", {}).get("backbone", "transformer")
    objective = cfg.get("train", {}).get("objective", "next_event")
    sources = cfg.get("sources", [["hcrl"]])
    targets = cfg.get("targets", [d for d in all_domains()])
    subsets = cfg.get("subsets", ["unseen", "seen"])
    cache: dict = {}
    rows = []
    for source in sources:
        src = _train(cfg, source, spec, backbone, objective, cache)
        source_ids = set(map(str, src.source_ids))
        target_cache: dict[str, pd.DataFrame] = {}
        for target in targets:
            if target in source:
                continue
            if target not in target_cache:
                target_cache[target] = _target_frames(cfg, target)
            frames = target_cache[target]
            if frames.empty:
                continue
            target_ids = set(frames["can_id"].astype(str).unique().tolist())
            for subset in subsets:
                sub = unseen_id_subset(frames, source_ids, mode=subset)
                if sub.empty:
                    continue
                res = evaluate_source_on_target(src, sub, length,
                                                fpr_target=float(cfg.get("fpr_target", 0.01)))
                base = _base_row(cfg, source, target, backbone, objective)
                base.update({
                    "subset": subset,
                    "n_source_ids": len(source_ids),
                    "n_target_ids": len(target_ids),
                    "n_shared_ids": len(source_ids & target_ids),
                })
                rows.append(_row_from_result(base, res, cfg))
                print(f"[phase4] cross-id {('+'.join(source))}->{target} [{subset}] "
                      f"shared={len(source_ids & target_ids)} f1={res['f1']:.3f} "
                      f"fpr={res['target_actual_fpr']:.4f} tpr@1%={res.get('tpr_at_0.01_fpr')}")
            del target_cache[target]  # free memory between targets
    df = pd.DataFrame(rows)
    save_csv(out / "cross_id_results.csv", df)
    return df


def run_ablation(cfg: dict, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(cfg.get("seed", 0)))
    base_spec = _spec(cfg)
    source = cfg.get("source", ["hcrl", "road", "sonata"])
    target = cfg.get("target", "kia_soul")
    rows = []
    grid = cfg.get("grid", {})
    cache: dict = {}
    for id_mode in grid.get("id_mode", [base_spec.id_mode]):
        for payload_mode in grid.get("payload_mode", [base_spec.payload_mode]):
            for length in grid.get("length", [cfg.get("length", 128)]):
                for backbone in grid.get(
                    "backbone", [cfg.get("model", {}).get("backbone", "transformer")]
                ):
                    for objective in grid.get(
                        "objective", [cfg.get("train", {}).get("objective", "next_event")]
                    ):
                        spec = TokenizerConfig(
                            payload_mode=payload_mode, id_mode=id_mode, id_bits=base_spec.id_bits
                        )
                        exp_cfg = dict(cfg)
                        exp_cfg["length"] = int(length)
                        exp_cfg["tokenizer"] = dict(
                            cfg.get("tokenizer", {}), id_mode=id_mode, payload_mode=payload_mode
                        )
                        tr = _train(exp_cfg, source, spec, backbone, objective, cache)
                        frames = _target_frames(exp_cfg, target)
                        res = evaluate_source_on_target(
                            tr, frames, int(length), fpr_target=float(cfg.get("fpr_target", 0.01))
                        )
                        base = _base_row(exp_cfg, source, target, backbone, objective)
                        rows.append(_row_from_result(base, res, cfg))
                        print(
                            f"[phase4] ablation id={id_mode} pay={payload_mode} L={length} "
                            f"{backbone}/{objective}: f1={res['f1']:.3f} "
                            f"fpr={res['target_actual_fpr']:.4f}"
                        )
    df = pd.DataFrame(rows)
    save_csv(out / "ablation_results.csv", df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 CAN-TFM runner")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["manifest", "prototype", "zero_shot", "cross_id", "ablation"],
        default="prototype",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.get("output_dir", "experiments/phase4/results"))
    if args.mode == "manifest":
        run_manifest(cfg, out)
    elif args.mode == "prototype":
        run_prototype(cfg, out)
    elif args.mode == "zero_shot":
        run_zero_shot(cfg, out)
    elif args.mode == "cross_id":
        run_cross_id(cfg, out)
    else:
        run_ablation(cfg, out)
    print(f"Phase 4 {args.mode} complete. Results under {out}")


if __name__ == "__main__":
    main()
