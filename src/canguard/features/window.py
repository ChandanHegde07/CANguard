
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

VALID_LABEL_POLICIES = ("any", "majority", "last")


class PerIDWindow:


    def __init__(self, maxlen: int) -> None:
        self.timestamps: deque[float] = deque(maxlen=maxlen)
        self.dlcs: deque[int] = deque(maxlen=maxlen)
        self.payloads: deque[list[int | None]] = deque(maxlen=maxlen)
        self.attack_flags: deque[int] = deque(maxlen=maxlen)

    def push(self, ts: float, dlc: int, data: list[str | None], is_attack: int = 0):
        self.timestamps.append(ts)
        self.dlcs.append(dlc)
        self.attack_flags.append(int(is_attack))
        int_data = [int(v, 16) if isinstance(v, str) and v.strip() else None for v in data]
        self.payloads.append(int_data)

    @property
    def full(self) -> bool:
        return len(self.timestamps) == self.timestamps.maxlen

    def iat_features(self) -> dict[str, float]:
        gaps = np.diff(list(self.timestamps))
        if len(gaps) < 2:
            return {
                "iat_mean": np.nan,
                "iat_std": np.nan,
                "iat_median": np.nan,
                "iat_min": np.nan,
                "iat_max": np.nan,
            }
        return {
            "iat_mean": float(np.mean(gaps)),
            "iat_std": float(np.std(gaps)),
            "iat_median": float(np.median(gaps)),
            "iat_min": float(np.min(gaps)),
            "iat_max": float(np.max(gaps)),
        }

    def dlc_features(self) -> dict[str, float]:
        a = np.array(list(self.dlcs))
        return {
            "dlc_mode": int(pd.Series(a).mode().iloc[0]) if len(a) else np.nan,
            "dlc_std": float(np.std(a)) if len(a) > 1 else 0.0,
        }

    def byte_features(self) -> dict[str, float]:
        arr = np.array(self.payloads, dtype=float)
        n = arr.shape[0]
        if n < 2:
            return {
                "byte_mean": np.nan,
                "byte_var": np.nan,
                "byte_max_change": np.nan,
                "byte_nunique": np.nan,
                "byte_entropy": np.nan,
            }
        bmean = float(np.nanmean(arr))
        bvar = float(np.nanvar(arr))
        diffs = np.abs(np.diff(arr, axis=0))
        maxchg = float(np.nanmax(diffs)) if diffs.size else np.nan
        flat = arr[~np.isnan(arr)].astype(int)
        nunique = float(len(np.unique(flat))) if len(flat) else np.nan
        entropy = np.nan
        if len(flat):
            counts = np.bincount(flat)
            probs = counts[counts > 0] / len(flat)
            entropy = float(-np.sum(probs * np.log2(probs)))
        return {
            "byte_mean": bmean,
            "byte_var": bvar,
            "byte_max_change": maxchg,
            "byte_nunique": nunique,
            "byte_entropy": entropy,
        }

    def compute_features(self) -> dict[str, float]:
        feats: dict[str, float] = {}
        feats.update(self.iat_features())
        feats.update(self.dlc_features())
        feats.update(self.byte_features())
        feats["window_fill"] = len(self.timestamps) / self.timestamps.maxlen
        return feats

    def compute_label(self, policy: str = "any") -> tuple[int, float]:
        flags = np.array(self.attack_flags)
        frac = float(flags.mean())
        if policy == "any":
            return int(flags.sum() > 0), frac
        if policy == "majority":
            return int(frac > 0.5), frac
        if policy == "last":
            return int(flags[-1]), frac
        raise ValueError(f"Unknown label_policy: {policy}")


class GlobalCANContext:
    """Cross-ID state: track last-seen timestamps and prune stale entries."""

    def __init__(self, decay_seconds: float = 1.0):
        self.last_seen: dict[str, float] = {}
        self.decay = decay_seconds

    def observe(self, can_id: str, ts: float) -> dict[str, float]:
        prev = self.last_seen.get(can_id)
        ts_last = prev if prev is not None else ts
        self.last_seen[can_id] = ts
        stale = [k for k, v in list(self.last_seen.items()) if ts - v > self.decay]
        for k in stale:
            del self.last_seen[k]
        return {"time_since_last_seen": ts - ts_last}

    def is_new_id(self, can_id: str, known_ids: set[str]) -> bool:
        return can_id not in known_ids


class FeaturePipeline:

    def __init__(
        self,
        window_size: int = 30,
        known_ids: set[str] | None = None,
        label_policy: str = "any",
        include_presence_features: bool = False,
    ) -> None:
        if label_policy not in VALID_LABEL_POLICIES:
            raise ValueError(f"label_policy must be one of {VALID_LABEL_POLICIES}")
        self.window_size = window_size
        self.windows: defaultdict[str, PerIDWindow] = defaultdict(lambda: PerIDWindow(window_size))
        self.ctx = GlobalCANContext()
        self.known_ids: set[str] = known_ids or set()
        self.label_policy = label_policy
        self.include_presence = include_presence_features

    def process_one(
        self,
        ts: float,
        can_id: str,
        dlc: int,
        data: list[str | None],
        is_attack: int = 0,
    ) -> dict | None:
        win = self.windows[can_id]
        win.push(ts, dlc, data, is_attack=is_attack)
        g = self.ctx.observe(can_id, ts)
        if not win.full:
            return None
        local_feats = win.compute_features()
        al, af = win.compute_label(policy=self.label_policy)
        feats: dict = {
            "can_id": can_id,
            "timestamp": ts,
            "dlc": dlc,
            **local_feats,
            "attack_frac": af,
            "is_attack": al,
        }
        feats["time_since_last_seen"] = g["time_since_last_seen"]
        if self.include_presence:
            feats["is_new_id"] = int(self.ctx.is_new_id(can_id, self.known_ids))
        return feats

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in df.sort_values("timestamp").iterrows():
            data = [row.get(f"data_{i}", None) for i in range(8)]
            attack = int(row.get("is_attack", 0))
            feats = self.process_one(
                row["timestamp"], row["can_id"], row["dlc"], data, is_attack=attack
            )
            if feats is not None:
                if "label" in df.columns:
                    feats["label"] = row["label"]
                records.append(feats)
        return pd.DataFrame(records)

    def reset(self) -> None:
        self.windows.clear()
        self.ctx = GlobalCANContext()


def fit_known_ids_on_normal_prefix(df: pd.DataFrame, n_normal: int = 20000) -> set[str]:
    return set(df[df["label"] == "R"].head(n_normal)["can_id"].unique())
