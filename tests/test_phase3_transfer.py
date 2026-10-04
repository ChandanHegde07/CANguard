"""Tests for Phase 3 source→target transfer infrastructure (leakage, IDs, metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canguard.detectors import IsolationForestDetector
from canguard.evaluation.transfer import (
    evaluate_transfer,
    fit_pird_source,
    split_captures,
    split_vehicles,
    validate_domain_schema,
    vehicle_domains,
)
from canguard.evaluation.transfer_metrics import (
    canonical_can_id,
    classify_window_ids,
    id_overlap,
    recall_at_fpr,
)
from canguard.evaluation.transfer_shift import feature_shift_table, summarize_shift
from canguard.features.groups import BEHAVIORAL_FEATURES_V1

FEATURES = list(BEHAVIORAL_FEATURES_V1)


def _domain(ids, n_per_id=60, seed=0, attack_ids=(), shift=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    ts = 0.0
    for cid in ids:
        for i in range(n_per_id):
            attack = 1 if (cid in attack_ids and i >= n_per_id // 2) else 0
            r = {
                "can_id": cid, "timestamp": ts, "is_attack": attack,
                "attack_frac": float(attack),
            }
            for f in FEATURES:
                r[f] = float(rng.normal(5.0 + shift, 1.0))
            if attack:
                r["iat_mean"] += 3.0
            rows.append(r)
            ts += 0.001
    return pd.DataFrame(rows)


def _source(ids=("0316", "018f"), **kw):
    df = _domain(ids, **kw)
    calib = df.iloc[: len(df) // 2]
    rest = df.iloc[len(df) // 2:]
    half = len(rest) // 2
    return fit_pird_source(calib, rest.iloc[:half], rest.iloc[half:], FEATURES,
                           IsolationForestDetector(n_estimators=20, random_state=0))


def test_canonical_can_id():
    assert canonical_can_id("0316") == "316"
    assert canonical_can_id("0x316") == "316"
    assert canonical_can_id("6E0") == "6e0"
    assert canonical_can_id("006") == "6"


def test_threshold_from_source_validation_only():
    src = _source()
    expected = float(np.percentile(src.val_normal_scores, 99.0))
    assert src.threshold == pytest.approx(expected)


def test_target_change_cannot_alter_source_model():
    src = _source()
    t1 = _domain(["316", "055"], n_per_id=60, seed=1)
    t2 = t1.copy()
    t2.loc[t2["is_attack"] == 0, FEATURES] = 1e6
    r1 = evaluate_transfer(src, t1, FEATURES)
    r2 = evaluate_transfer(src, t2, FEATURES)
    # source threshold is frozen regardless of target data
    assert r1["source_threshold"] == r2["source_threshold"]
    assert src.threshold == r1["source_threshold"]


def test_unseen_ids_are_reported_not_dropped():
    src = _source(ids=("0316",))
    target = _domain(["316", "055", "077"], n_per_id=60, seed=2)
    res = evaluate_transfer(src, target, FEATURES)
    assert res["n_known_id_windows"] + res["n_unseen_id_windows"] == len(target)
    assert res["n_shared_ids"] >= 1
    assert res["n_target_only_ids"] == 2
    assert 0.0 < res["target_unseen_id_window_fraction"] < 1.0


def test_primary_metrics_use_known_ids_only():
    src = _source(ids=("0316",))
    target = _domain(["316"], n_per_id=60, seed=3, attack_ids=("316",))
    res = evaluate_transfer(src, target, FEATURES)
    assert res["n_unseen_id_windows"] == 0
    assert np.isfinite(res["primary_recall"])


def test_id_overlap_counts_and_frame_weight():
    ids = {canonical_can_id(x) for x in ("0316", "018f")}
    tgt = {canonical_can_id(x) for x in ("316", "055")}
    counts = {"316": 90, "055": 10}
    ov = id_overlap(ids, tgt, counts)
    assert ov["n_shared_ids"] == 1
    assert ov["n_target_only_ids"] == 1
    assert ov["frame_weighted_overlap"] == pytest.approx(0.9)
    assert ov["id_count_overlap"] == pytest.approx(0.5)


def test_classify_window_ids():
    known = {"316", "18f"}
    mask = classify_window_ids(["316", "055", "18f"], known)
    assert mask.tolist() == [True, False, True]


def test_recall_at_fpr():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, 2000)
    scores = np.concatenate([rng.normal(0, 1, 900), rng.normal(5, 1, 100)])
    y = np.concatenate([np.zeros(900, int), np.ones(100, int)])
    out = recall_at_fpr(normal, y, scores, 0.01)
    assert out["actual_fpr"] < 0.03
    assert out["recall"] > 0.8


def test_transfer_determinism():
    src = _source()
    target = _domain(["316", "055"], n_per_id=60, seed=4, attack_ids=("316",))
    a = evaluate_transfer(src, target, FEATURES)
    b = evaluate_transfer(src, target, FEATURES)
    assert np.isfinite(a["primary_f1"])
    assert a["primary_f1"] == b["primary_f1"]
    assert a["target_actual_fpr"] == b["target_actual_fpr"]


def test_shift_smd_known_and_summary():
    s = _domain(["316"], n_per_id=80, seed=5)
    t = _domain(["316"], n_per_id=80, seed=6, shift=2.0)
    tbl = feature_shift_table(s, t, FEATURES)
    assert set(tbl.columns) >= {"feature", "smd", "abs_smd"}
    assert (tbl["abs_smd"].diff().dropna() <= 1e-9).all()  # ranked descending
    summ = summarize_shift(tbl)
    assert summ["max_abs_smd"] >= summ["median_abs_smd"]


# --- schema validation / capture- and vehicle-level splitting (Phase 3 §19) ---
def test_validate_domain_schema_accepts_and_rejects():
    good = _domain(["316"], n_per_id=25)
    validate_domain_schema(good, FEATURES)  # must not raise
    bad = good.drop(columns=["iat_mean"])
    with pytest.raises(ValueError, match="iat_mean"):
        validate_domain_schema(bad, FEATURES)
    no_id = good.drop(columns=["can_id"])
    with pytest.raises(ValueError, match="can_id"):
        validate_domain_schema(no_id, FEATURES)


def test_fit_source_rejects_incompatible_schema():
    good = _domain(["316"], n_per_id=25)
    calib = good.iloc[:10]
    with pytest.raises(ValueError):
        fit_pird_source(calib.drop(columns=["byte_var"]), calib, calib, FEATURES,
                        IsolationForestDetector(n_estimators=10, random_state=0))


def test_split_captures_is_disjoint_and_deterministic():
    caps = [f"capture_{i}" for i in range(10)]
    a = split_captures(caps, target_frac=0.3, seed=7)
    b = split_captures(caps, target_frac=0.3, seed=7)
    assert a == b  # deterministic given the seed
    assert set(a["source"]).isdisjoint(a["target"])
    assert set(a["source"]) | set(a["target"]) == set(caps)
    assert a["source"] and a["target"]
    with pytest.raises(ValueError):
        split_captures(["only_one"])


def test_split_vehicles_validates_and_disjoint():
    out = split_vehicles(["sonata", "kia", "spark"], ["sonata"], ["kia", "spark"])
    assert out["source"] == ["sonata"]
    assert out["target"] == ["kia", "spark"]
    with pytest.raises(ValueError):
        split_vehicles(["sonata", "kia"], ["sonata"], ["sonata"])
    with pytest.raises(ValueError):
        split_vehicles(["sonata"], ["sonata"], ["spark"])
    with pytest.raises(ValueError):
        split_vehicles(["sonata", "kia"], [], ["kia"])


def test_vehicle_domains_partition_by_vehicle():
    df = _domain(["316"], n_per_id=20)
    df["vehicle"] = ["sonata"] * 10 + ["kia"] * 10
    doms = vehicle_domains(df)
    assert set(doms) == {"sonata", "kia"}
    assert len(doms["sonata"]) == 10 and len(doms["kia"]) == 10
    with pytest.raises(ValueError):
        vehicle_domains(df.drop(columns=["vehicle"]))


def test_fit_source_does_not_mutate_inputs():
    calib = _domain(["316", "18f"], n_per_id=40, seed=9)
    if_fit = _domain(["316", "18f"], n_per_id=40, seed=10)
    val = _domain(["316", "18f"], n_per_id=40, seed=11)
    before = {n: d.copy(deep=True) for n, d in
              (("c", calib), ("i", if_fit), ("v", val))}
    fit_pird_source(calib, if_fit, val, FEATURES,
                    IsolationForestDetector(n_estimators=10, random_state=0))
    for name, d in (("c", calib), ("i", if_fit), ("v", val)):
        pd.testing.assert_frame_equal(d, before[name])
