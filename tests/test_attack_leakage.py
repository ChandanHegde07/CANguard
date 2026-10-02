"""Leakage-prevention tests for Phase 2 attack generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canguard.attacks.adaptive import AttackParameters, apply_fixed_attack
from canguard.features import fit_per_id_stats
from canguard.features.groups import BEHAVIORAL_FEATURES_V1

FEATURES = list(BEHAVIORAL_FEATURES_V1)


def _table(n_per_id: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ts = 0.0
    for cid in ("0316", "018f", "0000"):
        for i in range(n_per_id):
            attack = 1 if (cid == "0000" and i > n_per_id // 2) else 0
            r = {"can_id": cid, "timestamp": ts, "is_attack": attack, "attack_frac": float(attack)}
            r.update({f: float(rng.normal(5.0, 1.0)) for f in FEATURES})
            rows.append(r)
            ts += 0.001
    return pd.DataFrame(rows)


def test_fit_per_id_stats_ignores_attack_rows_in_calib():
    calib = _table()
    stats1, global1 = fit_per_id_stats(calib, FEATURES)
    perturbed = calib.copy()
    # Change attack rows drastically; stats must not move.
    atk_mask = perturbed["is_attack"] == 1
    perturbed.loc[atk_mask, FEATURES] = 1e6
    stats2, global2 = fit_per_id_stats(perturbed, FEATURES)
    assert set(stats1) == set(stats2)
    for cid in stats1:
        for f in FEATURES:
            assert stats1[cid][0][f] == pytest.approx(stats2[cid][0][f])
            assert stats1[cid][1][f] == pytest.approx(stats2[cid][1][f])
    for f in FEATURES:
        assert global1[0][f] == pytest.approx(global2[0][f])


def test_attack_parameters_cannot_be_built_from_test_provenance():
    calib = _table()
    stats, gstats = fit_per_id_stats(calib, FEATURES)
    with pytest.raises(ValueError):
        AttackParameters.from_stats(stats, gstats, FEATURES, provenance="test")
    with pytest.raises(ValueError):
        AttackParameters.from_stats(stats, gstats, FEATURES, provenance="test_attack")


def test_generated_attacks_depend_only_on_frozen_params():
    calib = _table()
    stats, gstats = fit_per_id_stats(calib, FEATURES)
    params = AttackParameters.from_stats(stats, gstats, FEATURES, provenance="calib_normal")
    substrate = _table(seed=99)
    substrate = substrate[substrate["can_id"] == "0316"].reset_index(drop=True)

    out1 = apply_fixed_attack(params, substrate, "0316", 1.0, "positive", FEATURES,
                              mode="additive")
    # Simulate an extreme test stream arriving later; frozen params must not change.
    _ = _table(seed=1234)
    out2 = apply_fixed_attack(params, substrate, "0316", 1.0, "positive", FEATURES,
                              mode="additive")
    pd.testing.assert_frame_equal(out1, out2)


def test_target_selection_uses_calibration_labels_only():
    from experiments.runners.run_phase2 import select_legit_targets

    calib = _table()
    stats, _ = fit_per_id_stats(calib, FEATURES)
    # 0000 has attack windows in calibration -> excluded.
    targets = select_legit_targets(calib, stats, top_k=5, min_calib_windows=20,
                                   max_calib_attack_frac=0.0)
    assert "0000" not in targets
    assert "0316" in targets and "018f" in targets
