# Phase 2 — Adversarial Robustness, Global Extension, and External Validation

**Definitive Phase 2 report.** PIRD is frozen throughout. This document
consolidates: (i) the white-box adaptive-attacker study, (ii) the global /
cross-ID extension and hybrid, (iii) the normal-distribution-shift diagnosis,
and (iv) frozen external validation on ROAD.

Reproduce (each command is self-contained):

```bash
python -m experiments.runners.run_phase2 --config experiments/configs/phase2_baseline.yaml --mode baseline
python -m experiments.runners.run_phase2 --config experiments/configs/phase2_adaptive.yaml --mode adaptive
python -m experiments.runners.run_phase2 --config experiments/configs/phase2_gradual.yaml  --mode gradual
python -m experiments.runners.run_phase2 --config experiments/configs/phase2_hybrid.yaml   --mode hybrid
python -m experiments.runners.run_phase2 --config experiments/configs/phase2_global_shift.yaml --mode shift
python -m experiments.runners.run_phase2 --config experiments/configs/phase2_road_external.yaml --mode road
```

---

## 1. Research questions

1. Does PIRD generalize beyond the dataset it was designed on?
2. How does PIRD behave against a white-box adaptive attacker?
3. What is its empirical detection boundary?
4. Can global behavioral statistics detect attacks outside PIRD's local per-ID representation?
5. Does the global layer complement PIRD?
6. Does the hybrid remain useful under realistic FPR constraints?
7. Do these findings reproduce on ROAD?
8. Why does the global representation experience normal-distribution shift?

---

## 2. Frozen PIRD baseline

```
CAN frames → FeaturePipeline(ws=30) → 14 BEHAVIORAL_FEATURES_V1
→ temporal 40/20/40 split → calibration normals → per-ID μ_k, σ_k
→ r = (x − μ_k)/(σ_k + ε) → Isolation Forest (200 trees, seed 0)
→ threshold = 1%-FPR percentile on held-out train normals → test
```

| Dataset | F1 | Precision | Recall | FPR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| DoS | 0.017 | 0.056 | 0.010 | 0.055 | 0.894 | 0.630 |
| Fuzzy | 0.473 | 0.312 | 0.978 | 0.144 | 0.973 | 0.759 |
| RPM | 0.992 | 0.984 | 0.999 | 0.004 | 0.999 | 0.991 |
| gear | 0.946 | 0.899 | 0.998 | 0.032 | 0.999 | 0.996 |

PIRD is the frozen control branch in every experiment below.

---

## 3. White-box adaptive attacker

The attacker knows the CAN IDs, the 14-feature representation, the per-ID
statistics `(μ_k, σ_k)`, the residual transform, the detector and its threshold.
It compromises an existing legitimate ID and sets

```
mean_shift :  x_attack = μ_k + α · σ_k · direction
additive   :  x_attack = x      + α · σ_k · direction
```

clipped to feasible feature ranges and rounded for discrete features; `α = 0` is
a no-op. All attacker knowledge comes from calibration normals only; provenance
guards reject any non-calibration statistics. A matched unmodified control and a
one-sided two-proportion test (p<0.05) are used for every attack condition.

---

## 4. Adaptive attack results (HCRL)

Window-level detection probability, and control-adjusted significant detection:

| α | PIRD det | PIRD sig | Control (target IDs) |
|---|---|---|---|
| 0.25 | 0.057 | 0.188 | 0.088 |
| 0.50 | 0.131 | 0.375 | 0.088 |
| 0.75 | 0.190 | 0.417 | 0.088 |
| 1.00 | 0.239 | 0.438 | 0.088 |
| 1.50 | 0.435 | 0.625 | 0.088 |
| 2.00 | 0.689 | 0.813 | 0.088 |
| 3.00 | 0.809 | 0.875 | 0.088 |

* Pooled boundary: **α₅₀ ≈ 1.63**, α₉₀ **not reached** within the grid
  (`ceiling_not_reached`).
* `additive` (modify-existing-message) is the stealthier parameterization:
  control-adjusted **α₅₀ ≈ 0.38, α₉₀ ≈ 1.80**.
* Strong ID heterogeneity: control-adjusted α₅₀ ranges from ≈0.58 (Fuzzy `0316`)
  to not reached (RPM `0153`); DoS `018f` already alarms on ~84% of unmodified
  windows, so only the control-adjusted measure is meaningful there.
* Gradual drift (slow, exponential): control-adjusted detection 0.06 (α_end 0.25)
  → 0.79 (α_end 3.0); **α₅₀ ≈ 0.89**, α₉₀ not reached.

A legitimate ID can carry a full 1σ systematic shift on 13 features and still be
caught on only ~24% of windows.

---

## 5. Global behavioral extension

28 causal / window-causal features over a trailing 200-message window:

* **Bus (14)**: frame count/rate, IAT mean/std/cv/min/max/median, burstiness, DLC
  mean/std, `g_bus_bytes_per_sec_proxy` (occupancy proxy), payload mean.
* **ID-population (9)**: unique/active/new IDs, `H(ID) = -Σ p_k log p_k`,
  normalized entropy, concentration, dominant / top-3 fractions, IDs above 5%.
* **Cross-ID (5)**: per-ID count CV, frequency L1 / cosine distance to
  calibration, payload and IAT dispersion across IDs.

Normalization is global (robust median/IQR), fit on calibration normals only.
Detector = Isolation Forest (200 trees, seed 0), same threshold protocol. Hybrid
rules: OR, max, weighted-λ (λ ∈ {0.25, 0.5, 0.75}), on validation-normal
z-scored branch scores.

---

## 6. Global / hybrid results (HCRL, 1% target)

| Dataset | PIRD F1 | Global F1 | Global FPR | Hybrid max F1 | Hybrid max FPR |
|---|---|---|---|---|---|
| DoS | 0.017 | **0.495** | 0.666 | 0.494 | 0.668 |
| Fuzzy | **0.473** | 0.237 | 0.352 | 0.272 | 0.336 |
| RPM | **0.992** | 0.015 | 0.009 | 0.985 | 0.008 |
| gear | **0.946** | 0.011 | 0.003 | 0.945 | 0.032 |

Global detects population-level attacks (DoS, Fuzzy) but not per-ID spoofing
(RPM, gear), and at a large FPR. Hybrid `max` preserves PIRD on RPM/gear while
inheriting global's DoS behaviour and FPR; weighted rules trade Fuzzy/RPM for
DoS recall. On adaptive attacks the global branch is at chance
(control-adjusted significant = **0.0** at every α 0.25–3.0) and the hybrid does
**not** recover PIRD's blind spot (e.g. at α=1.0: PIRD 0.438 vs max 0.250,
OR 0.375, weighted-0.5 0.604 but with a 24% control alarm rate vs PIRD's 8.8%).
Gradual drift: global significant = 0.0 at all endpoints; PIRD is the most
sensitive branch.

**Fixed-FPR (pooled mean across datasets; actual test FPR shown):**

| Rule | recall @ actual FPR |
|---|---|
| PIRD | 0.301 @ 1.6%, 0.746 @ 5.9%, 0.749 @ 9.4% |
| Global | 0.456 @ 23.9%, 0.464 @ 25.7%, 0.488 @ 29.0% |
| max | 0.555 @ 17.0%, 0.987 @ 26.1%, 0.999 @ 29.5% |
| weighted-0.75 | 0.593 @ 7.9%, 0.916 @ 17.6%, 1.000 @ 21.7% |

The global branch cannot be operated at a low FPR under this calibration
protocol.

---

## 7. Complementarity

Detection overlap at the 1% branch thresholds (all test windows):

| Dataset | PIRD only | Global only | Both | Neither | Score corr. (Pearson) |
|---|---|---|---|---|---|
| DoS | 199 | 16882 | 841 | 5770 | 0.579 |
| Fuzzy | 493 | 4343 | 3570 | 12289 | 0.523 |
| RPM | 4971 | 159 | 39 | 18536 | 0.102 |
| gear | 5746 | 50 | 34 | 17872 | −0.138 |

The branches are genuinely different detectors (near-zero / negative correlation
on RPM/gear). However, global's "unique" detections on RPM/gear have precision
≈0.01–0.19; they are almost entirely false positives. **Complementary signals
are not necessarily useful when the additional signal is dominated by false
positives.** On DoS/Fuzzy the complementary signal is real but inseparable from
the normal-distribution shift (below).

---

## 8. False-positive / distribution-shift analysis

Only normal traffic is used here (calibration normal vs test normal).

**Score shift (the direct cause of the FPR):**

| Dataset | val-normal mean | test-normal mean | threshold | actual FPR |
|---|---|---|---|---|
| DoS | 0.393 | 0.493 | 0.445 | 0.666 |
| Fuzzy | 0.383 | 0.454 | 0.475 | 0.352 |
| RPM | 0.513 | 0.501 | 0.631 | 0.009 |
| gear | 0.542 | 0.510 | 0.632 | 0.003 |

For DoS/Fuzzy the test-normal score distribution shifts above the threshold
calibrated on validation normals; for RPM/gear it does not. The shift is a
calibration/regime mismatch, not an attack.

**Feature-level shift (ranked by |SMD|; `global_shift/feature_shift.csv`):**

| Dataset | Strongest features (|SMD|) | Family pattern (mean |SMD|) |
|---|---|---|
| DoS | `g_n_unique_ids`/`g_active_id_ratio` (~0.49), `g_iat_std_across_ids` (0.28), `g_iat_cv` (0.25) | id_pop 0.168 > bus 0.153 > cross 0.127 |
| Fuzzy | `g_payload_std_across_ids` (**1.71**), `g_payload_bus_mean` (1.29), `g_n_unique_ids` (0.76), `g_id_entropy` (0.59) | cross 0.620 > id_pop 0.493 > bus 0.258 |
| RPM | `g_n_unique_ids`/`g_active_id_ratio` (0.60), `g_iat_max` (−0.47), `g_iat_cv` (−0.40) | bus 0.205 > id_pop 0.186 > cross 0.168 |
| gear | `g_n_unique_ids`/`g_active_id_ratio` (0.51), `g_iat_std` (−0.50), `g_iat_mean` (−0.44), `g_frame_rate` (0.42) | bus 0.315 > cross 0.189 > id_pop 0.170 |

The shift is **diffuse across all three families**, not attributable to a single
feature. The most consistent signature is ID activity/population
(`g_n_unique_ids`, `g_active_id_ratio`) shifting up in the test segment of every
dataset, plus timing/payload shifts that differ per dataset.

**Shift ↔ false-alarm association (test normals only; `flag_association.csv`):**

| Dataset | Top associated features (|SMD| flagged vs not-flagged, Spearman with score) |
|---|---|
| DoS | `g_iat_max` (1.46, ρ=0.71), `g_n_ids_above_5pct` (1.32, ρ=−0.81), `g_new_id_count` (1.26, ρ=0.79), `g_freq_l1_to_calib` (1.25, ρ=0.74) |
| Fuzzy | `g_n_ids_above_5pct` (**2.59**, ρ=−0.80), `g_frame_rate` (2.10, ρ=−0.70), `g_freq_l1_to_calib` (1.70) |
| RPM | `g_dlc_mean` (1.72), `g_dlc_std` (1.56) (few flagged) |
| gear | `g_dlc_mean` (0.79), `g_dlc_std` (0.76) (few flagged) |

Flagged test-normal windows have systematically more extreme ID-population,
cross-ID and timing features than non-flagged normal windows, consistent with the
feature-shift table. This is an **association**, not a causal feature attribution
(the Isolation Forest is not interpretable at the split level).

**Diagnostic calibration-coverage experiment** (labelled; not the primary
frozen result). Broadening the global calibration from calibration-only normals
to calibration+train normals:

| Dataset | primary FPR → broad FPR | primary recall → broad recall |
|---|---|---|
| DoS | 0.666 → **0.000** | 1.000 → **0.000** |
| Fuzzy | 0.352 → **0.012** | 0.843 → **0.035** |
| RPM | 0.009 → 0.012 | 0.008 → 0.010 |
| gear | 0.003 → 0.006 | 0.006 → 0.012 |

Broadening normal calibration eliminates the false positives **and** the
DoS/Fuzzy detections together. This shows the global branch's population-attack
"detection" is largely the calibration/test regime shift itself; the shift is not
a fixable calibration-coverage artifact alone — it is non-stationarity between
the calibration and test segments of these captures.

---

## 9. ROAD external validation

_Frozen per-capture pre-injection protocol. PIRD / global / hybrid parameters are
identical to the HCRL experiments; only the data split follows ROAD's documented
protocol (calibration = pre-injection normals; test = post-injection traffic).
No ROAD attack/test observation enters calibration._

## 9. ROAD external validation

_Frozen per-capture pre-injection protocol. PIRD / global / hybrid parameters are
identical to the HCRL experiments; only the data split follows ROAD's documented
protocol: calibration = pre-injection normals, test = post-injection traffic. No
ROAD attack/test observation enters calibration. One eligible capture per attack
type (masquerade variants excluded), full captures (no sampling)._

### 9.1 Conventional ROAD attacks (operating point at the frozen 1% target)

F1 (recall, actual test FPR):

| ROAD attack | PIRD | Global | Hybrid max |
|---|---|---|---|
| correlated_signal | **0.445** (0.999, 0.192) | 0.146 (0.987, 0.890) | 0.188 (0.999, 0.668) |
| fuzzing | **0.206** (0.174, 0.192) | 0.144 (0.153, 0.358) | 0.168 (0.177, 0.341) |
| max_engine_coolant_temp | 0.266 (1.000, 0.041) | 0.000 (0.000, 0.012) | **0.331** (1.000, 0.030) |
| max_speedometer | **0.743** (0.998, 0.031) | 0.094 (0.131, 0.075) | 0.596 (0.998, 0.062) |
| reverse_light_off | **0.711** (0.980, 0.040) | 0.142 (0.967, 0.606) | 0.202 (0.980, 0.400) |
| reverse_light_on | **0.626** (0.700, 0.026) | 0.122 (0.869, 0.592) | 0.188 (0.815, 0.328) |
| **Normal FPR** (mean) | **~0.09** | ~0.42 | ~0.30 |

PIRD is the best branch on every targeted ROAD attack. Global reaches high recall
on correlated_signal/reverse_light but at 0.59–0.89 FPR (precision 0.07–0.08);
hybrid `max` inherits that FPR. The one place hybrid `max` helps is
`max_engine_coolant_temp` (F1 0.331 vs 0.266), a single capture with only 102
attack windows, so this is weak evidence.

### 9.2 Fixed-FPR (pooled mean across ROAD captures; actual test FPR)

| Rule | recall @ actual FPR |
|---|---|
| PIRD | 0.201 @ 1.0%, 0.809 @ 8.7%, 0.880 @ 23.8% |
| Global | 0.400 @ 26.9%, 0.518 @ 42.2%, 0.682 @ 59.4% |
| max | 0.228 @ 3.2%, 0.828 @ 30.5%, 0.905 @ 57.9% |
| weighted-0.75 | 0.733 @ 5.2%, 0.879 @ 16.6%, 0.908 @ 37.9% |

The global branch again cannot be operated at low FPR on independent data; PIRD
has the best recall at any comparable FPR.

### 9.3 ROAD adaptive attack (frozen, matched control)

Applied to post-injection **normal** windows exactly as on HCRL. Control-adjusted
boundary:

| Rule | α₅₀ (significant) | α₉₀ (significant) | mean control alarm rate |
|---|---|---|---|
| PIRD | 1.80 | not reached | 0.038 |
| Global | not reached (0.0 at all α) | not reached | 0.408 |
| max | 1.97 | not reached | 0.285 |
| OR | 1.83 | not reached | 0.415 |
| weighted-0.5 | 1.03 | 1.48 | 0.233 |
| weighted-0.75 | 1.32 | 2.56 | 0.415 |

Global contributes **no** attack-attributable adaptive detection on ROAD either.
PIRD's control-adjusted α₅₀ (1.80) is higher than on HCRL (1.17) but remains the
only meaningful adaptive detector; the weighted hybrids' lower α₅₀ values are
paid for by 6–11× higher control alarm rates.

### 9.4 Does ROAD confirm the HCRL conclusions?

**Yes.** On independent data: (i) PIRD is the best branch for per-ID attacks;
(ii) the global branch suffers a severe normal-distribution shift and high FPR
(0.59–0.89 on several captures); (iii) hybrid `max` does not improve on PIRD and
inherits global's FPR; (iv) global contributes zero attack-attributable adaptive
detection. The only deviation is a marginal hybrid gain on a single
low-attack-count capture, which is not statistically strong.

_Note:_ the current frozen loader/protocol yields PIRD ROAD FPRs of 0.03–0.19,
higher than some values in the older committed `tables/road_baseline_results.csv`;
those tables were produced with a different frame-selection version. This
validation matches the current `prepare_road_capture` protocol exactly.

<!-- ROAD_RESULTS_END -->

---

## 10. Limitations

* **Feature-space adaptive threat.** The adaptive attacker modifies per-ID
  window feature vectors; it is an upper bound on an attacker who can realise
  such features, not a raw-frame-level attack.
* **Dataset / vehicle diversity.** HCRL is one vehicle/bus; ROAD adds a second
  corpus but only the attack scenarios present there. Cross-vehicle
  generalisation beyond these two corpora is untested.
* **Calibration requirements.** Global (and PIRD) calibration assumes a
  representative pre-deployment / pre-injection normal segment; HCRL's
  calibration/test regime shift violates this and is the direct cause of the
  global FPR.
* **Adaptive attacker assumptions.** White-box knowledge of `(μ_k, σ_k)` and the
  detector is a worst case; a black-box attacker would be weaker.
* **Global representation shift.** The 28 features shift diffusely across bus,
  ID-population and cross-ID families; no single family or feature explains the
  FPR.
* **Hybrid FPR.** No hybrid rule recovers the adaptive blind spot at an
  acceptable FPR; weighted rules trade precision for recall.
* **Attack-scenario coverage.** Gradual drift was not transferred to ROAD in this
  pass (synthetic result); ROAD fuzzing mixes cross-ID effects with the loader's
  labelling policy.
* **No robust-global redesign.** Deliberately out of scope: the task is
  diagnosis, and the global model was not optimised.

---

## 11. Final scientific conclusions

### Supported findings
* PIRD is a strong detector for the **studied per-ID behavioural attacks**
  (RPM/gear recall ≈ 1.0) and recovers them on ROAD.
* A white-box attacker that constrains manipulation to remain close to the
  learned per-ID distribution **substantially reduces PIRD detection**: pooled
  detection 5.7% (α=0.25) to 80.9% (α=3.0), α₅₀ ≈ 1.63; the stealthier additive
  parameterization reaches α₅₀ ≈ 0.38.
* Global bus/population statistics are **sensitive to population-level attacks**
  (DoS recall 1.0) and carry information weakly/negatively correlated with PIRD
  on per-ID spoofing data.
* The global branch suffers a **severe normal-distribution shift** (target FPR
  1% → realized 24–67% on DoS/Fuzzy), caused by a diffuse calibration/test regime
  mismatch, not by one feature.
* These conclusions **reproduce on independent ROAD data**: PIRD is best for
  per-ID attacks; global FPR is 0.59–0.89 on several captures; hybrid `max` does
  not beat PIRD; global adaptive detection is 0.0 at every severity.

### Partially supported findings
* The global layer **complements** PIRD in representation (different score
  structure) but is only useful on population attacks, and only at a high FPR.
* The hybrid `max` preserves PIRD performance on per-ID attacks while adding
  population-attack recall, but inherits the global FPR and does not improve
  adaptive detection.

### Unsupported hypotheses
* That the global layer would recover PIRD's adaptive-attack blind spot: it does
  not (control-adjusted significance 0.0 at every severity).
* That the hybrid is deployable under realistic FPR constraints: the global
  branch cannot be operated at low FPR under the current calibration protocol.
* That weak PIRD/global correlation implies a useful hybrid: much of the global
  branch's unique detection is false-positive behaviour.

### Open questions
* Robust/online global modelling under non-stationary normal traffic.
* Higher-fidelity raw-frame adaptive attacks and physical realizability.
* Cross-vehicle validation beyond HCRL/ROAD, and ROAD gradual-drift transfer.
* Whether a global branch calibrated on a genuinely stationary normal regime can
  add complementary signal at an acceptable FPR.

---
