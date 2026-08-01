# ROAD Dataset Validation & HCRL Comparison

This document records the Phase 10 validation: does Per-ID Behavioral Residual
Detection (PIRD) generalize from HCRL to ROAD?

PIRD itself is frozen. Nothing in the feature, residual, or detector logic was
changed. Only a data loader and experiment runner were added.

---

## Dataset differences (HCRL vs ROAD)

| Property | HCRL Car-Hacking | ROAD |
|----------|------------------|------|
| Vehicle | Hyundai YF Sonata (synthetic, scripted) | Real vehicle on dynamometer (ORNL) |
| Frame format | `timestamp,CAN_ID,DLC,data[0..7],label` | `(ts) can0 HEXID#HEXDATA` (DLC=8 only) |
| CAN ID ext | 11-bit standard IDs | 11-bit standard IDs |
| Attack structure | One isolated attack type per file, uniform cadence | Verified real attacks with per-capture injection intervals |
| Attack types | DoS, fuzzing, RPM spoof, gear spoof | correlated-signal, fuzzing, max-coolant-temp, max-speedometer, reverse-light off/on |
| Label basis | Per-row class label in CSV | Interval+ID matching against capture metadata |
| Temporal structure | Single continuous stream per attack file | Independent driving-session captures |
| Variable DLC | Yes (DLC<8 rows present) | No (always DLC=8) |
| Presence-based | Yes — trivial new-ID/constant-byte separation | Attacks reuse legitimate AIDs on the same bus |

**Key structural difference**: ROAD captures are independent driving sessions.
Temporal continuity does not span captures, so PIRD's single-stream
40/20/40 temporal split cannot be applied to a concatenated multi-capture table.
The correct application is per-capture: per-ID residuals are fitted on the
pre-injection normal frames of each capture.

---

## Method

For each attack capture (excluding `accelerator_*` which have no injected
message, and `*_masquerade` duplicates):

1. Parse the capture with `RoadLoader` (interval+ID attack labeling).
2. Build per-ID windows with the unchanged `FeaturePipeline` (window=30).
3. Fit per-ID normal stats on capture normals (`fit_per_id_stats`).
4. Residualize (`transform_residuals`), split chronologically 40/20/40.
5. `IsolationForestDetector` (200 trees) on train normals; threshold ~1% FPR
   on held-out val normals.
6. Score the full capture; compute precision/recall/F1/ROC-AUC/PR-AUC.

---

## Results (mean over captures per attack type)

| Attack type | F1 | Recall | FPR | ROC-AUC |
|-------------|-----|--------|-----|---------|
| correlated_signal | 0.898 | 1.000 | 0.020 | ~0.99 |
| fuzzing | 0.273 | 0.340 | 0.015 | moderate |
| max_engine_coolant_temp | 0.265 | 1.000 | 0.024 | ~0.99 |
| max_speedometer | 0.698 | 1.000 | 0.038 | ~0.99 |
| reverse_light_off | 0.657 | 1.000 | 0.048 | ~0.99 |
| reverse_light_on | 0.544 | 0.997 | 0.053 | ~0.99 |

Per-capture detail is in `results/road/road_results.json`.

---

## Research questions

**Does PIRD generalize beyond HCRL?**
Partially. PIRD achieves **recall ≈ 1.0** on most ROAD attacks (correlated
signal, max-coolant-temp, max-speedometer, reverse-light on/off), which is a
strong generalization result given those attacks reuse legitimate AIDs on the
same bus. However, FPR is elevated (2–5%) relative to the 1% HCRL target, and
F1 is dragged down by that FPR and by low attack density within captures.

**Which attacks improve vs degrade?**
* Relative to HCRL, targeted single-AID injections (speedometer, correlated
  signal, reverse-light) generalize well — PIRD detects the injected period.
  This is a genuine improvement over HCRL's DoS failure.
* **fuzzing degrades** (recall 0.34, F1 0.27). ROAD fuzzing is more subtle than
  HCRL fuzzing: it floods many AIDs with the injected value but within a short
  window, so per-ID residual baselines mostly absorb it and the temporal split
  lands the injection partly in train.

**Are behavioral assumptions still valid?**
Yes for cadence/signal attacks. Per-ID IAT + byte behavior discriminates the
injected AID even when the ID is legitimate. The assumption breaks for fuzzing,
where injection is a brief burst across all IDs — a per-ID (not cross-ID) norm.

**Which attack types violate PIRD assumptions?**
Fuzzing (cross-ID injection, not per-ID) and to a lesser degree long low-density
injections where the low FPR budget is spent before the attack appears.

**How does ROAD differ from HCRL and why does it matter?**
ROAD attacks are real and reuse legitimate AIDs; the bus is noisier (real
driving) and captures are short/session-based. This is why presence-based
shortcuts (`is_new_id`) would fail on ROAD, and why per-capture evaluation is
required. PIRD's per-ID residuals are the right signal, but its global ~1% FPR
budget is stressed by benign driving-state variation.

---

## Takeaway

PIRD is a **valid behavioral IDS methodology** that transfers to ROAD for
targeted single-AID attacks (high recall, near-perfect ROC/PR-AUC). It does
*not* fully generalize to cross-ID fuzzing with the frozen per-ID configuration.
These are validation findings, not defects to tune away.
