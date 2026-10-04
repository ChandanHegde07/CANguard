"""Domain/capture loading and windowing for Phase 4.

A *domain* is a dataset (``hcrl``, ``road``, ``gem_can``) or a Survival vehicle
(``sonata``, ``kia_soul``, ``chevrolet_spark``). Frames are placed in the
canonical schema used across the repository and then encoded by the Phase 4
tokenizer. Windows never cross capture boundaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from canguard.data.gem_can import GemCanLoader
from canguard.data.survival import SurvivalLoader
from canguard.evaluation.transfer_metrics import canonicalize_frame_ids
from canguard.exp.cache import FeatureCache
from canguard.exp.road_protocol import list_eval_captures, load_capture_frames, resolve_road_root

from .tokenizer import TokenizerConfig, frames_to_features, make_windows

SURVIVAL_VEHICLES = ("sonata", "kia_soul", "chevrolet_spark")
HCRL_FILES = ("DoS", "Fuzzy", "RPM", "gear")
DATASET_DOMAINS = ("hcrl", "road", "gem_can")

DEFAULT_ROOTS = {
    "hcrl": "data",
    "road": "road/road",
    "gem_can": "GEM_CAN_Dataset_R2/GEM_CAN_Dataset_R2",
    "survival": "survival",
}


def all_domains() -> list[str]:
    return list(DATASET_DOMAINS) + list(SURVIVAL_VEHICLES)


def domain_family(domain: str) -> str:
    if domain in DATASET_DOMAINS:
        return domain
    if domain in SURVIVAL_VEHICLES:
        return "survival"
    raise ValueError(f"unknown domain {domain!r}; known: {all_domains()}")


def _roots(roots: dict | None) -> dict:
    r = dict(DEFAULT_ROOTS)
    if roots:
        r.update(roots)
    return r


def list_captures(domain: str, roots: dict | None = None) -> list[str]:
    r = _roots(roots)
    fam = domain_family(domain)
    if fam == "hcrl":
        return [f for f in HCRL_FILES if (Path(r["hcrl"]) / f"{f}_dataset.csv").exists()]
    if fam == "road":
        return [
            name
            for name, _ in list_eval_captures(
                resolve_road_root(r["road"]), skip_masquerade=True, skip_unlabeled=True
            )
        ]
    if fam == "gem_can":
        return GemCanLoader(r["gem_can"]).list_captures()
    return [stem for stem, _, _ in SurvivalLoader(r["survival"]).list_captures(domain)]


def capture_path(domain: str, capture: str, roots: dict | None = None) -> Path:
    r = _roots(roots)
    fam = domain_family(domain)
    if fam == "hcrl":
        return Path(r["hcrl"]) / f"{capture}_dataset.csv"
    if fam == "gem_can":
        return GemCanLoader(r["gem_can"]).capture_path(capture)
    if fam == "survival":
        return SurvivalLoader(r["survival"]).capture_path(domain, capture)
    return resolve_road_root(r["road"]) / "attacks" / f"{capture}.log"


def _finalize(df: pd.DataFrame, domain: str, capture: str, attack_type: str | None) -> pd.DataFrame:
    df = df.copy()
    df["can_id"] = canonicalize_frame_ids(df["can_id"])
    if attack_type is not None and "attack_type" not in df.columns:
        df["attack_type"] = np.where(df["is_attack"] == 1, attack_type, "Normal")
    elif "attack_type" not in df.columns:
        df["attack_type"] = np.where(df["is_attack"] == 1, capture, "Normal")
    df["domain"] = domain
    df["capture"] = capture
    df["is_attack"] = df["is_attack"].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _load_hcrl_fast(path: Path, max_frames: int | None, tail: bool) -> pd.DataFrame:
    """Read a HCRL CSV with pandas, optionally only a head/tail slice.

    HCRL rows have a **variable** number of fields: the payload holds exactly
    ``dlc`` bytes and the label sits at column ``3 + dlc`` (not the last column).
    Pandas pads short rows with NaN, so the label must be gathered per row.
    """
    total = _count_lines(path) if max_frames is not None else None
    kwargs = dict(header=None, names=list(range(12)), engine="python")
    if max_frames is None or total is None or total <= max_frames:
        raw = pd.read_csv(path, **kwargs)
    elif tail:
        raw = pd.read_csv(path, skiprows=max(0, total - max_frames), **kwargs)
    else:
        raw = pd.read_csv(path, nrows=max_frames, **kwargs)
    arr = raw.to_numpy(dtype=object)
    dlc = pd.to_numeric(pd.Series(arr[:, 2]), errors="coerce").fillna(0).astype(int).to_numpy()
    rows = np.arange(len(arr))
    label_idx = np.clip(3 + dlc, 0, arr.shape[1] - 1)
    labels = arr[rows, label_idx]
    data = np.full((len(arr), 8), None, dtype=object)
    for i in range(8):
        m = i < dlc
        data[m, i] = arr[m, 3 + i]
    out = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(pd.Series(arr[:, 0]), errors="coerce"),
            "can_id": pd.Series(arr[:, 1]).astype(str),
            "dlc": dlc,
            "label": pd.Series(labels).astype(str),
            **{f"data_{i}": data[:, i] for i in range(8)},
        }
    )
    out["is_attack"] = (out["label"].str.strip() != "R").astype(int)
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def load_capture_frames_domain(
    domain: str,
    capture: str,
    roots: dict | None = None,
    max_frames: int | None = None,
    tail: bool = False,
    cache: FeatureCache | None = None,
) -> pd.DataFrame:
    """Load one capture in the canonical schema (labels + attack_type), cached."""
    r = _roots(roots)
    fam = domain_family(domain)
    if cache is not None:
        try:
            mtime = capture_path(domain, capture, r).stat().st_mtime
        except OSError:
            mtime = 0
        key = {
            "stage": "phase4_frames_v2",
            "domain": domain,
            "capture": capture,
            "max_frames": max_frames,
            "tail": tail,
            "mtime": mtime,
        }
        hit = cache.load_df(key)
        if hit is not None:
            return hit

    if fam == "hcrl":
        df = _load_hcrl_fast(capture_path(domain, capture, r), max_frames, tail)
        out = _finalize(df, domain, capture, attack_type=capture)
    elif fam == "road":
        root = resolve_road_root(r["road"])
        meta = next(m for n, m in list_eval_captures(root, True, True) if n == capture)
        df = load_capture_frames(root, capture, meta, max_frames=max_frames)
        out = _finalize(df, domain, capture, attack_type=None)
    elif fam == "gem_can":
        df = GemCanLoader(r["gem_can"]).load_capture(capture)
        if max_frames is not None and len(df) > max_frames:
            df = df.iloc[-max_frames:] if tail else df.iloc[:max_frames]
        out = _finalize(df, domain, capture, attack_type=None)
    else:
        df = SurvivalLoader(r["survival"]).load_capture(domain, capture)
        if max_frames is not None and len(df) > max_frames:
            df = df.iloc[-max_frames:] if tail else df.iloc[:max_frames]
        out = _finalize(df, domain, capture, attack_type=None)

    if cache is not None:
        cache.save_df(key, out)
    return out


def load_domain_frames(
    domain: str,
    roots: dict | None = None,
    max_frames_per_capture: int | None = None,
    tail_hcrl: bool = True,
    captures: list[str] | None = None,
    cache: FeatureCache | None = None,
) -> pd.DataFrame:
    """Concatenate all (or selected) captures of a domain."""
    parts = []
    for cap in captures or list_captures(domain, roots):
        df = load_capture_frames_domain(
            domain,
            cap,
            roots,
            max_frames=max_frames_per_capture,
            tail=tail_hcrl and domain_family(domain) == "hcrl",
            cache=cache,
        )
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_windows_per_capture(
    df: pd.DataFrame, spec: TokenizerConfig, length: int, stride: int
) -> dict:
    """Build non-crossing windows for every capture in ``df``.

    Returns dict with ``X [M,L,F]``, ``y [M,L]``, ``valid [M,L,8]``,
    ``frame_row [M,L]`` (row indices into ``df``), ``starts`` (global first
    frame of each window) and per-window ``capture``.
    """
    df = df.reset_index(drop=True)
    cap_values = df["capture"].to_numpy()
    Xs, ys, Vs, rows, starts, caps = [], [], [], [], [], []
    for cap in pd.unique(cap_values):
        pos = np.flatnonzero(cap_values == cap)
        g = df.iloc[pos]
        feats, _slices, valid = frames_to_features(g, spec)
        if len(feats) == 0:
            continue
        labels = g["is_attack"].to_numpy(dtype=np.int64)
        X, y, st = make_windows(feats, labels, length, stride)
        if len(X) == 0:
            continue
        V = np.stack([valid[s : s + length] for s in st])
        Xs.append(X)
        ys.append(y)
        Vs.append(V)
        rows.append(np.stack([pos[s : s + length] for s in st]))
        starts.append(pos[st])
        caps.extend([str(cap)] * len(st))
    if not Xs:
        empty = np.zeros((0, length, spec.feature_dim), np.float32)
        return {
            "X": empty,
            "y": np.zeros((0, length), np.int64),
            "valid": np.zeros((0, length, 8), bool),
            "frame_row": np.zeros((0, length), np.int64),
            "starts": np.zeros(0, np.int64),
            "capture": [],
        }
    return {
        "X": np.concatenate(Xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "valid": np.concatenate(Vs, axis=0),
        "frame_row": np.concatenate(rows, axis=0).astype(np.int64),
        "starts": np.concatenate(starts).astype(np.int64),
        "capture": np.asarray(caps, dtype=object),
    }


def dataset_manifest(
    domains: list[str], roots: dict | None = None, max_frames_per_capture: int | None = None
) -> list[dict]:
    """Provenance records (one per domain/capture)."""
    records = []
    for domain in domains:
        for cap in list_captures(domain, roots):
            try:
                df = load_capture_frames_domain(
                    domain, cap, roots, max_frames=max_frames_per_capture
                )
            except Exception as exc:  # pragma: no cover - surfaced in manifest
                records.append(
                    {"domain": domain, "capture": cap, "available": False, "error": str(exc)}
                )
                continue
            ids = df["can_id"].map(lambda s: int(str(s), 16) if str(s) else 0)
            records.append(
                {
                    "dataset": domain_family(domain),
                    "domain": domain,
                    "vehicle": domain if domain in SURVIVAL_VEHICLES else "",
                    "capture": cap,
                    "available": True,
                    "n_frames": int(len(df)),
                    "n_ids": int(df["can_id"].nunique()),
                    "bit_width": int(max(11, ids.max().bit_length() if len(ids) else 11)),
                    "can_id_format": "11bit" if domain != "gem_can" else "29bit_extended",
                    "n_normal": int((df["is_attack"] == 0).sum()),
                    "n_attack": int((df["is_attack"] == 1).sum()),
                    "attack_types": sorted(
                        df.loc[df["is_attack"] == 1, "attack_type"].unique().tolist()
                    ),
                }
            )
    return records


__all__ = [
    "DEFAULT_ROOTS",
    "DATASET_DOMAINS",
    "SURVIVAL_VEHICLES",
    "all_domains",
    "build_windows_per_capture",
    "capture_path",
    "dataset_manifest",
    "domain_family",
    "list_captures",
    "load_capture_frames_domain",
    "load_domain_frames",
]
