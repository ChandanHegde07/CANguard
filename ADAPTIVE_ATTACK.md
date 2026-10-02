# ADAPTIVE_ATTACK.md — White-Box Adaptive Attacks Against PIRD (Phase 2)

This document specifies the threat model, mathematical formulation, leakage
controls, experimental protocol, and results of the first Phase 2 study. PIRD
itself is **unchanged**; Phase 2 is a control experiment that characterizes where
PIRD breaks under an attacker who knows its internals.

---

## 1. Threat model

The attacker is **white-box** with respect to the detector. It knows:

* the set of CAN arbitration IDs and which ones are legitimate;
* the PIRD feature representation (the 14 `BEHAVIORAL_FEATURES_V1`);
* the per-ID normal statistics `(mu_k, sigma_k)` fitted on calibration normals;
* the residual transform `r = (x - mu_k) / (sigma_k + EPS)`;
* the detector (Isolation Forest) and its frozen threshold.

The attacker can **compromise an existing legitimate ID** and control the
behavioral feature vector of that ID's messages (e.g., payload values and/or
timing). It does **not** need to introduce a novel ID. The goal is to introduce
malicious manipulation while remaining statistically consistent with normal
behavior.

### What is assumed away (limitations)

The attack is a **feature-space** attack and therefore an *upper bound* on an
attacker who can realize arbitrary feature vectors. Independent manipulation of
correlated features (e.g., `iat_mean`, `iat_median`, `iat_max`) is not guaranteed
to be physically realizable from a single payload/timing change; mapping a target
`alpha` to real CAN-frame modifications is left to future work. Bounds and
integer rounding keep every generated window physically valid, but they do not
enforce cross-feature joint consistency.

---

## 2. Mathematical formulation

For a target legitimate ID `k`, the attacker replaces the feature vector `x` by

```
x_attack = mu_k + alpha * sigma_k * direction        (mean_shift mode)
x_attack = x        + alpha * sigma_k * direction        (additive mode)
```

per targeted feature, then clips each feature to its feasible range and rounds
discrete features. `direction ∈ {+1, -1}` is the attack direction and `alpha` is
the severity in per-ID standard-deviation units.

* With the linear residual transform, `mean_shift` yields `r ≈ alpha * direction`
  (a deterministic center shift on the targeted features).
* `additive` yields `r ≈ z_normal + alpha * direction`, i.e. the natural
  per-window variation is preserved and shifted. This models *modifying an
  existing message* rather than replacing it with the population mean.

`alpha = 0` is defined to be a no-op in both modes.

Feasible ranges: `byte_mean ∈ [0,255]`, `byte_var ∈ [0,255²]`,
`byte_max_change ∈ [0,255]`, `byte_nunique ∈ [0,240]` (integer),
`byte_entropy ∈ [0,8]`, `dlc_mode ∈ {0..8}` (integer), `dlc_std ∈ [0,8]`,
IAT / `time_since_last_seen ≥ 0`, `window_fill ∈ [0,1]`. Negative directions on
non-negative features saturate at the boundary, so their residual magnitude
plateaus — this is reported, not hidden.

---

## 3. Leakage prevention

```
CALIB normals ──► fit (mu, sigma) ──┐
TRAIN normals ──► fit IsolationForest ──┼──► FREEZE ──► generate attack ──► TEST normals
TRAIN val normals ──► threshold ────────┘
```

* Per-ID means/stds come only from `calib[is_attack == 0]`
  (`fit_per_id_stats`).
* The detector is fit only on train normals; the threshold is the
  `(1 - 0.01)`-percentile of scores on held-out train normals
  (`choose_threshold_from_val_normals`). Both are frozen before any attack is
  generated.
* `AttackParameters.from_stats` refuses any provenance other than
  `"calib_normal"`.
* Target IDs are selected using calibration labels/statistics only
  (`select_legit_targets`): an ID needs ≥ `min_calib_windows` calibration-normal
  windows and an attack fraction in calibration of 0.
* Attack generation reads only `(mu, sigma)` and the substrate values; it never
  reads test labels, test scores, or statistics estimated from test data.
* The injection substrate is the **test-normal** windows (`is_attack == 0`).
  Real attack windows are never used to fit or threshold anything.

Unit tests assert all of the above (`tests/test_attack_leakage.py`,
`tests/test_attack_adaptive.py`, `tests/test_phase2_runner.py`).

---

## 4. Experimental protocol

Frozen PIRD configuration (reproduces the repository baseline):

| Setting | Value |
|---|---|
| Dataset | HCRL Car-Hacking (`DoS`, `Fuzzy`, `RPM`, `gear`) |
| Sample | first 60,000 frames per file |
| Window | `window_size = 30`, per-ID sliding window |
| Features | 14 `BEHAVIORAL_FEATURES_V1` |
| Split | temporal 40% calib / 20% train / 40% test |
| Residual | per-ID z-score from calibration normals, global fallback |
| Detector | Isolation Forest, 200 trees, `random_state=0` |
| Threshold | percentile at 1% FPR on held-out train normals |
| Seed | 0 |

Pre-registered attack grid (defined before running, not adjusted afterwards):

* severities `alpha ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}`
* directions `{positive, negative}`
* modes `{mean_shift, additive}`
* target IDs: top-3 legitimate IDs by calibration window count, per dataset
* gradual drift: exponential ramp, rates `{fast=0.25, medium=0.5, slow=1.0}` of a
  300-window horizon, end-severity from the same grid

Evaluation for a target ID:
* all of the target's test-normal windows are overwritten (fixed sweep) or a
  contiguous 300-window block is ramped (gradual);
* untouched test normals remain label 0, overwritten windows are label 1;
* detection probability = fraction of injected windows scored ≥ threshold;
* a **matched control** is the same target windows *unmodified* — used to
  compute each ID's pre-existing alarm rate and a control-adjusted detection
  test (one-sided two-proportion z-test, `p < 0.05`).

### Baseline (frozen)

| Dataset | F1 | Precision | Recall | FPR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| DoS | 0.017 | 0.056 | 0.010 | 0.055 | 0.894 | 0.630 |
| Fuzzy | 0.473 | 0.312 | 0.978 | 0.144 | 0.973 | 0.759 |
| RPM | 0.992 | 0.984 | 0.999 | 0.004 | 0.999 | 0.991 |
| gear | 0.946 | 0.899 | 0.998 | 0.032 | 0.999 | 0.996 |

These match the repository's published numbers (README). The real HCRL
injections are the "non-adaptive" attack: near-perfect recall on RPM/gear,
failure on DoS.

---

## 5. Results

### 5.1 Fixed-severity detection boundary (pooled)

`detection_probability` = fraction of injected windows flagged. Because some
legitimate IDs already alarm frequently, the **control-adjusted** column
(significant elevation over the same ID's unmodified windows) is also given.

| α | Detection prob (window) | 95% CI | Control-adjusted | Mean target control FPR | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.057 | [0.055, 0.059] | 0.188 | 0.088 | 0.054 | 0.057 | 0.055 |
| 0.50 | 0.131 | [0.128, 0.134] | 0.375 | 0.088 | 0.115 | 0.131 | 0.122 |
| 0.75 | 0.190 | [0.187, 0.194] | 0.417 | 0.088 | 0.159 | 0.190 | 0.173 |
| 1.00 | 0.239 | [0.236, 0.243] | 0.438 | 0.088 | 0.192 | 0.239 | 0.213 |
| 1.50 | 0.435 | [0.431, 0.440] | 0.625 | 0.088 | 0.302 | 0.435 | 0.357 |
| 2.00 | 0.689 | [0.685, 0.693] | 0.813 | 0.088 | 0.406 | 0.689 | 0.511 |
| 3.00 | 0.809 | [0.805, 0.812] | 0.875 | 0.088 | 0.445 | 0.809 | 0.574 |

Operational boundary (pooled):

| Definition | α₅₀ | α₉₀ |
|---|---|---|
| Window detection probability | **1.63** | not reached (ceiling) |
| Control-adjusted detection | **1.17** | not reached (ceiling) |

By attack parameterization (control-adjusted α₅₀ / α₉₀):

| Mode | α₅₀ | α₉₀ |
|---|---|---|
| additive (modify existing message) | **0.375** | **1.80** |
| mean_shift (literal equation) | 1.69 | not reached |

`mean_shift` produces a near-identical residual vector for every window of an
ID, so its per-config detection is essentially all-or-nothing (step function).
`additive` preserves natural variation and gives a smooth, statistically
meaningful curve; it is the more realistic and sensitive threat.

### 5.2 Heterogeneity across legitimate IDs

The boundary varies enormously by ID and dataset. Representative control-adjusted
α₅₀ (additive mode; `None` = 50% never reached within the grid):

| Dataset | Target | direction | α₅₀ |
|---|---|---|---|
| Fuzzy | 0316 | positive | 0.58 |
| Fuzzy | 02c0 | positive | 0.60 |
| gear | 0260 | negative | 1.04 |
| DoS | 0316 | positive | 1.45 |
| RPM | 018f | positive | 2.00 |
| RPM | 0153 | negative | None (≤ 0.08 at α=3) |
| DoS | 018f | either | None (already ~84% control alarm rate) |

Two distinct regimes appear:

1. **Low-baseline IDs** (e.g. RPM 0153, Fuzzy 0153): PIRD almost never alarms on
   them normally, so even small attacks are statistically detectable, but the
   *window-level* detection rate remains low until α is large.
2. **Noisy IDs** (e.g. DoS 018f): the ID already alarms on ~84% of normal
   windows, so high naive "detection" carries no attack-attributable signal and
   the control-adjusted measure is the only meaningful one.

No ID was dropped; heterogeneity is reported in
`adaptive_attack/severity_sweep.csv` (per dataset, target, direction, mode) and
`boundary_table_by_direction.csv`.

### 5.3 Gradual / slow-drift attack

Exponential ramp over a 300-window horizon, end severity `alpha_end`.

| α_end | Naive "any alarm" | Control-adjusted | Median delay (detected, windows) |
|---|---|---|---|
| 0.25 | 0.667 | 0.181 | fast 32 / medium 32 / slow 32 |
| 0.50 | 0.681 | 0.292 | 51 / 32 / 32 |
| 0.75 | 0.722 | 0.431 | 54 / 51 / 51 |
| 1.00 | 0.778 | 0.556 | 51 / 67 / 51 |
| 1.50 | 0.847 | 0.694 | 44 / 71 / 75 |
| 2.00 | 0.847 | 0.764 | 35 / 54 / 75 |
| 3.00 | 0.889 | 0.861 | 25 / 49 / 79 |

The naive "any alarm in 300 windows" is **dominated by the per-window
false-alarm rate** (~5.8%): `1 - (1 - 0.058)^300 ≈ 1`. It is therefore reported
only for completeness; the control-adjusted curve is the meaningful result. Slow
ramps are detected later and less often than fast ramps, and the 50% crossing for
control-adjusted gradual detection is `alpha_end ≈ 0.89`; 90% is not reached
within the grid.

### 5.4 Adaptive vs non-adaptive comparison

| Attack | Detection rate |
|---|---|
| Normal traffic | FPR 0.055 (DoS) … 0.144 (Fuzzy) |
| Non-adaptive (real HCRL) | recall 0.010 (DoS), 0.978 (Fuzzy), 0.999 (RPM), 0.998 (gear) |
| White-box adaptive, α=1.0 | window detection 0.239 (additive 0.356, mean_shift 0.122) |
| White-box adaptive, α=2.0 | window detection 0.689 |

**The adaptive attacker gains a large advantage.** The real HCRL injections are
presence-based and are caught almost perfectly on RPM/gear (recall ≈ 1.0), while
a white-box attacker who shifts a legitimate ID by one standard deviation on the
targeted features is caught on only ~24% of windows (pooled). In other words,
knowledge of PIRD's per-ID assumptions lets an attacker who *reuses a legitimate
ID* stay below the detector's operating threshold far more easily than the
dataset's own attacks do.

---

## 6. Conclusions (first pass)

1. PIRD is **highly evadable** by a white-box feature-space attacker. A
   systematic shift of up to one per-ID standard deviation on 13 features of a
   legitimate ID produces a pooled window-level detection probability of only
   ~24% (additive ~36%); 50% detection requires α ≈ 1.5–1.6.
2. The detection boundary is **strongly ID-dependent** and is confounded by
   per-ID baseline alarm rates. Control-adjusted analysis is essential; two IDs
   in the top-3 selections already alarm ≥ 80% of normal windows.
3. **Gradual drift** is harder to detect than a fixed jump for the same end
   severity, and slow ramps are the hardest. The naive "any alarm" metric is
   unusable without a false-alarm control.
4. No clean α₉₀ boundary exists within the pre-registered grid for the pooled
   window-level metric (`ceiling_not_reached`); we report this rather than
   extrapolating.
5. The result is an **upper bound** (feature-space attack). It establishes that
   PIRD's residual decision boundary leaves a wide statistically-normal band,
   motivating the hybrid/global extensions deferred to later Phase 2 work.

---

## 7. Reproduce

```bash
# 1. Frozen PIRD baseline (single command)
python -m experiments.runners.run_phase2 \
    --config experiments/configs/phase2_baseline.yaml --mode baseline

# 2. Fixed-severity white-box adaptive sweep + detection boundary
python -m experiments.runners.run_phase2 \
    --config experiments/configs/phase2_adaptive.yaml --mode adaptive

# 3. Gradual / slow-drift attack + latency
python -m experiments.runners.run_phase2 \
    --config experiments/configs/phase2_gradual.yaml --mode gradual

# tests
pytest tests/test_attack_adaptive.py tests/test_attack_gradual.py \
       tests/test_attack_leakage.py tests/test_boundary.py \
       tests/test_phase2_runner.py
```

Artifacts are written under `experiments/phase2/` (see `PHASE2_PLAN.md` for the
layout). Every run records `experiment_id`, dataset, split, seed, target IDs,
attack type, severity, drift rate, detector, window size, normalization and
threshold in the per-experiment `config.json` and `summary.json`.
