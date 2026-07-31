"""Equivalence tests for the feature pipeline vs notebook implementation."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from canguard.features import FeaturePipeline, GlobalCANContext, PerIDWindow


# ---------------------------------------------------------------------------
# Reference implementation (verbatim from feature_eng_hcrl.ipynb)
# ---------------------------------------------------------------------------
class _RefPerIDWindow:
    def __init__(self, maxlen: int):
        self.timestamps = deque(maxlen=maxlen)
        self.dlcs = deque(maxlen=maxlen)
        self.payloads = deque(maxlen=maxlen)

    def push(self, ts, dlc, data):
        self.timestamps.append(ts)
        self.dlcs.append(dlc)
        int_data = [int(v, 16) if isinstance(v, str) and v.strip() else None for v in data]
        self.payloads.append(int_data)

    @property
    def full(self):
        return len(self.timestamps) == self.timestamps.maxlen

    def iat_features(self):
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

    def dlc_features(self):
        a = np.array(list(self.dlcs))
        return {
            "dlc_mode": int(pd.Series(a).mode().iloc[0]) if len(a) else np.nan,
            "dlc_std": float(np.std(a)) if len(a) > 1 else 0.0,
        }

    def byte_features(self):
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
        if len(flat):
            counts = np.bincount(flat)
            probs = counts[counts > 0] / len(flat)
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = np.nan
        return {
            "byte_mean": bmean,
            "byte_var": bvar,
            "byte_max_change": maxchg,
            "byte_nunique": nunique,
            "byte_entropy": entropy,
        }

    def compute(self, label_policy="any"):
        feats = {}
        feats.update(self.iat_features())
        feats.update(self.dlc_features())
        feats.update(self.byte_features())
        feats["window_fill"] = len(self.timestamps) / self.timestamps.maxlen
        return feats


def _ref_pipeline(df, window_size=30, known_ids=None):
    windows = defaultdict(lambda: _RefPerIDWindow(window_size))
    last_seen = {}
    records = []
    for _, row in df.sort_values("timestamp").iterrows():
        cid = row["can_id"]
        ts = row["timestamp"]
        dlc = row["dlc"]
        data = [row.get(f"data_{i}", None) for i in range(8)]
        win = windows[cid]
        win.push(ts, dlc, data)
        ts_last = last_seen.get(cid, ts)
        last_seen[cid] = ts
        if not win.full:
            continue
        l = win.compute()
        feats = {
            "can_id": cid,
            "timestamp": ts,
            "dlc": dlc,
            **l,
            "time_since_last_seen": ts - ts_last,
        }
        if "label" in df.columns:
            feats["label"] = row["label"]
        if "is_attack" in df.columns:
            feats["is_attack"] = int(row["is_attack"])
        records.append(feats)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _synthetic_df(n_frames=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = ["0316", "018f", "02b0"]
    rows = []
    ts = 1478191030.0
    for i in range(n_frames):
        cid = ids[rng.integers(0, len(ids))]
        dlc = 8
        data = [f"{rng.integers(0, 255):02x}" for _ in range(8)]
        label = "T" if rng.random() < 0.1 else "R"
        row = {"timestamp": ts, "can_id": cid, "dlc": dlc, "label": label}
        row.update({f"data_{k}": data[k] for k in range(8)})
        rows.append(row)
        ts += rng.uniform(0.0002, 0.002)
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].astype(float)
    df["dlc"] = df["dlc"].astype(int)
    df["is_attack"] = (df["label"] != "R").astype(int)
    return df


def _identical_feature_rows(a, b):
    cols = sorted(c for c in a.columns if c not in ("can_id", "timestamp", "label"))
    a = a[sorted(cols)].reset_index(drop=True)
    b = b[sorted(cols)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_iat_features_matching():
    win = PerIDWindow(maxlen=50)
    ref = _RefPerIDWindow(maxlen=50)
    ts = np.linspace(0, 1, 50)
    for t in ts:
        win.push(float(t), 8, ["01"] * 8)
        ref.push(float(t), 8, ["01"] * 8)
    assert win.iat_features() == ref.iat_features()


def test_dlc_features_matching():
    win = PerIDWindow(maxlen=50)
    ref = _RefPerIDWindow(maxlen=50)
    for d in [8, 8, 5, 8, 5, 8, 8, 8]:
        win.push(0.0, d, ["01"] * d)
        ref.push(0.0, d, ["01"] * d)
    a, b = win.dlc_features(), ref.dlc_features()
    assert a["dlc_mode"] == b["dlc_mode"]
    assert abs(a["dlc_std"] - b["dlc_std"]) < 1e-9


def test_byte_features_matching():
    win = PerIDWindow(maxlen=50)
    ref = _RefPerIDWindow(maxlen=50)
    rng = np.random.default_rng(1)
    for _ in range(30):
        b = [f"{rng.integers(0, 255):02x}" for _ in range(8)]
        win.push(0.0, 8, b)
        ref.push(0.0, 8, b)
    a, b = win.byte_features(), ref.byte_features()
    for k in a:
        assert abs(a[k] - b[k]) < 1e-9, f"{k}: {a[k]} != {b[k]}"


def test_entropy_drops_on_constant_payload():
    win = PerIDWindow(maxlen=50)
    for _ in range(30):
        win.push(0.0, 8, ["24"] * 8)  # constant byte
    feats = win.byte_features()
    assert feats["byte_nunique"] == 1
    assert feats["byte_entropy"] == 0.0


def test_label_policies():
    win = PerIDWindow(maxlen=4)
    for a, t in [(0, 0.0), (1, 0.1), (0, 0.2), (0, 0.3)]:
        win.push(t, 8, ["00"] * 8, is_attack=a)
    assert win.compute_label("any") == (1, 0.25)
    assert win.compute_label("majority") == (0, 0.25)
    assert win.compute_label("last") == (0, 0.25)


def test_prune_global_context():
    ctx = GlobalCANContext(decay_seconds=1.0)
    for cid, t_ in [("a", 100.0), ("b", 100.5), ("c", 101.5)]:
        ctx.observe(cid, t_)
    # 'a' seen at 100.0 is stale when ts=101.5 (>1.0 s ago) -> pruned
    assert "a" not in ctx.last_seen
    # 'b' seen at 100.5 is exactly 1.0 s ago; pruning is strict '> decay' so it survives
    assert "b" in ctx.last_seen
    assert "c" in ctx.last_seen


def test_module_pipeline_matches_notebook_reference():
    df = _synthetic_df(400)
    have = FeaturePipeline(window_size=30).process_dataframe(df)
    expected = _ref_pipeline(df, window_size=30)

    # Module adds per-window label field `attack_frac` (and a window-policy
    # `is_attack`). The reference carries the raw-row is_attack directly.
    assert "attack_frac" in have.columns
    assert "is_attack" in expected.columns  # raw-row label carried through
    module_only = set(have.columns) - set(expected.columns)
    assert module_only == {"attack_frac"}

    # Compare only the shared feature columns that both derived from window
    # statistics (exclude labels since window-policy vs raw-row may differ).
    shared = [c for c in expected.columns if c in have.columns and c not in ("is_attack",)]
    a = have[shared].reset_index(drop=True)
    b = expected[shared].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_known_ids_prefix():
    df = _synthetic_df(100)
    from canguard.features.window import fit_known_ids_on_normal_prefix

    known = fit_known_ids_on_normal_prefix(df)
    assert isinstance(known, set)
    assert all(k != "0000" for k in known)  # synthetic has no novel ids injected
