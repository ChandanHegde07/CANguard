# PHASE2_HYBRID_REPORT.md — Global / Cross-ID Behavioral Layer + PIRD Hybrid

**Phase 2 extension.** Frozen PIRD is one branch; an independently-defined,
causal global/cross-ID branch is added and evaluated alone and in hybrid. The
question is whether global bus-level information complements PIRD, i.e. detects
anomalies that PIRD's per-ID residualization cannot, at an acceptable false
positive rate.

Reproduce:

```bash
python -m experiments.runners.run_phase2 \
    --config experiments/configs/phase2_hybrid.yaml --mode hybrid
```

The original `baseline/`, `adaptive_attack/`, `gradual_drift/` results are
untouched; new outputs live in `experiments/phase2/{global,hybrid,plots,results}/`.

---

## 1. Objective

PIRD asks *“is this CAN ID behaving abnormally relative to its own history?”*
That is powerful for per-ID value manipulation, but blind to changes in the
**population structure** of the bus (flooding, new IDs, frequency shifts,
cross-ID relationships). A global layer is introduced to test whether
population-level behaviour carries complementary signal. The global feature set
was defined from CAN-bus semantics **before** looking at any adaptive-attack
result, and the adaptive attacks were not modified.

---

## 2. Method

### 2.1 Global feature families (14 bus + 9 ID-population + 5 cross-ID = 28)

All features are **causal / window-causal**: the vector at row `t` is a function
of the trailing `window_frames=200` emitted messages at rows `<= t` only
(rolling counts and rolling moments). No future data is used.

| Family | Features |
|---|---|
| **Bus** (`A`) | `g_n_frames`, `g_duration_s`, `g_frame_rate`, `g_iat_mean/std/cv/min/max/median`, `g_burstiness`, `g_dlc_mean/std`, `g_bus_bytes_per_sec_proxy`, `g_payload_bus_mean` |
| **ID population** (`B`) | `g_n_unique_ids`, `g_new_id_count`, `g_id_entropy`, `g_id_entropy_norm`, `g_id_concentration`, `g_dominant_id_frac`, `g_top3_id_frac`, `g_n_ids_above_5pct`, `g_active_id_ratio` |
| **Cross-ID** (`C`) | `g_id_count_cv`, `g_freq_l1_to_calib`, `g_id_dist_cosine_to_calib`, `g_payload_std_across_ids`, `g_iat_std_across_ids` |

`H(ID) = -sum_k p_k log p_k` over per-ID message fractions `p_k` in the window;
`g_id_entropy_norm = H / log(n_unique)`; `g_id_concentration = 1 - g_id_entropy_norm`.
`g_bus_bytes_per_sec_proxy` is an explicitly named **occupancy proxy**
(sum of DLC / window duration), not a physical bus-load calculation.

This is a deliberately small, interpretable set — not feature explosion.
Causality per feature is recorded in `canguard.features.FEATURE_CAUSALITY`.
Features marked `causal_requires_calibration` (`g_new_id_count`,
`g_active_id_ratio`, `g_freq_l1_to_calib`, `g_id_dist_cosine_to_calib`) use a
frozen calibration ID set/distribution but never look forward in time.

### 2.2 Global normalization and detector

`GlobalStats` is fit on **calibration normals only** with either `zscore`
(mean/std) or `robust` (median / IQR·1.349). Feature residuals are
`r = (x - center)/(scale + EPS)` (global, **not** per-ID). The detector is the
repository's `IsolationForestDetector` (200 trees, seed 0) fit on train normals
with the same 1%-FPR threshold protocol. `normalization: robust` is the default.

### 2.3 Hybrid combination

Branch scores are z-normalized with **validation-normal** statistics only
(`ScoreNormalizer`). Rules:
* **OR** — flag if either branch exceeds its own threshold.
* **max** — `max(z_local, z_global)`.
* **weighted-λ** — `λ·z_local + (1−λ)·z_global`, λ ∈ {0.25, 0.5, 0.75}
  (pre-defined grid, never tuned on attacks).

Thresholds for max/weighted are chosen on val-normal scores at the target FPR.

---

## 3. Leakage controls

* Per-ID stats (`fit_per_id_stats`) and global stats (`fit_global_stats`) use
  `calib[is_attack == 0]` only; provenance guards reject non-calibration input.
* Both branch detectors and thresholds are frozen before any attack is generated.
* The global calibration ID set/distribution comes from calibration normals.
* Adaptive attacks reuse `canguard.attacks` unchanged; the attacked table is
  rebuilt from the frozen split. No test statistic enters calibration.
* Unit tests assert: attack rows do not change global stats; scaling/transform
  is calibration-only; changing a **future** row does not change earlier global
  features; the seeded runner is deterministic.

---

## 4. Experimental protocol

| Setting | Value |
|---|---|
| Dataset | HCRL Car-Hacking (`DoS`, `Fuzzy`, `RPM`, `gear`) |
| Sample | first 60,000 frames per file |
| Window | `window_size=30` per-ID sliding window (PIRD); `window_frames=200` global |
| Split | temporal 40% calib / 20% train / 40% test |
| Detector | Isolation Forest, 200 trees, `random_state=0` |
| Threshold | percentile at 1% FPR on held-out train normals |
| Seed | 0 |
| Attacks | unchanged Phase 2 grid: α ∈ {0.25,0.5,0.75,1.0,1.5,2.0,3.0}, directions ±, modes {additive, mean_shift}; gradual exponential fast/medium/slow |
| Normalization | global robust (median/IQR) |

Matched controls and the two-proportion significance test reuse the Phase 2
utilities (`detection_probability`, `_two_proportion_z`).

---

## 5. PIRD baseline (frozen reference)

| Dataset | F1 | Precision | Recall | FPR |
|---|---|---|---|---|
| DoS | 0.017 | 0.056 | 0.010 | 0.055 |
| Fuzzy | 0.473 | 0.312 | 0.978 | 0.144 |
| RPM | 0.992 | 0.984 | 0.999 | 0.004 |
| gear | 0.946 | 0.899 | 0.998 | 0.032 |

---

## 6. Global-only results

Conventional (1% target FPR on val normals):

| Dataset | Global F1 | Global recall | Global actual FPR | PIRD F1 |
|---|---|---|---|---|
| DoS | **0.495** | **1.000** | 0.666 | 0.017 |
| Fuzzy | 0.237 | 0.843 | 0.352 | 0.473 |
| RPM | 0.015 | 0.008 | 0.009 | 0.992 |
| gear | 0.011 | 0.006 | 0.003 | 0.946 |

Global alone detects population-level attacks (DoS flooding, Fuzzy injection)
but **not** the per-ID spoofing attacks (RPM/gear). Critically, it does so at
very high FPR (66% / 35%). On adaptive legitimate-ID attacks the global branch
is at chance: detection ≈ 0.242 vs matched-control 0.253, with
`detection_probability_significant = 0.0` at **every** severity (α 0.25–3.0).

---

## 7. Hybrid results (conventional, 1% target)

| Dataset | rule | recall | precision | F1 | actual FPR |
|---|---|---|---|---|---|
| DoS | PIRD | 0.010 | 0.056 | 0.017 | 0.055 |
| DoS | global | 1.000 | 0.329 | 0.495 | 0.666 |
| DoS | max | 1.000 | 0.328 | 0.494 | 0.668 |
| DoS | weighted_0.75 | 1.000 | 0.488 | **0.656** | 0.342 |
| Fuzzy | PIRD | 0.978 | 0.312 | 0.473 | 0.144 |
| Fuzzy | max | 0.951 | 0.159 | 0.272 | 0.336 |
| Fuzzy | weighted_0.75 | 0.999 | 0.167 | 0.286 | 0.334 |
| RPM | PIRD | 0.999 | 0.984 | 0.992 | 0.004 |
| RPM | max | 0.999 | 0.971 | 0.985 | 0.008 |
| RPM | weighted_0.5 | 0.282 | 0.896 | 0.429 | 0.009 |
| gear | PIRD | 0.998 | 0.899 | 0.946 | 0.032 |
| gear | max | 0.998 | 0.898 | 0.945 | 0.032 |
| gear | weighted_0.75 | 0.998 | 0.918 | 0.956 | 0.025 |

Findings: `max` preserves PIRD performance on RPM/gear and adds DoS recall, but
inherits global's FPR. Weighted rules degrade Fuzzy/RPM in exchange for DoS
recall. No hybrid rule improves on PIRD without a large FPR increase.

---

## 8. Adaptive attack results (control-adjusted)

Detection probability (fraction of injected windows flagged) and control-adjusted
significance (fraction of configs whose attack alarm rate exceeds their matched
unmodified control, one-sided two-proportion p<0.05):

| α | PIRD det | PIRD sig | Global det | Global sig | max det | max sig | or det | or sig | w0.5 sig |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.057 | 0.188 | 0.242 | **0.00** | 0.246 | 0.125 | 0.257 | 0.146 | 0.208 |
| 0.50 | 0.131 | 0.375 | 0.242 | **0.00** | 0.252 | 0.188 | 0.308 | 0.292 | 0.396 |
| 0.75 | 0.190 | 0.417 | 0.242 | **0.00** | 0.271 | 0.250 | 0.357 | 0.396 | 0.521 |
| 1.00 | 0.239 | 0.438 | 0.243 | **0.00** | 0.299 | 0.250 | 0.399 | 0.375 | 0.604 |
| 1.50 | 0.435 | 0.625 | 0.243 | **0.00** | 0.487 | 0.500 | 0.575 | 0.625 | 0.750 |
| 2.00 | 0.689 | 0.813 | 0.243 | **0.00** | 0.666 | 0.667 | 0.752 | 0.813 | 0.875 |
| 3.00 | 0.809 | 0.875 | 0.243 | **0.00** | 0.780 | 0.771 | 0.839 | 0.875 | 0.979 |

* **Global contributes nothing to adaptive detection**: it never achieves a
  statistically significant attack-specific lift at any severity (α 0.25–3.0).
  Its ~24% detection is its own false-alarm level (control 25.3%).
* **Hybrid does not recover the PIRD blind spot.** At α ≤ 1.0 the hybrid
  rules have control-adjusted detection *equal to or below* PIRD (e.g. at
  α=1.0: PIRD 0.438 vs max 0.250, OR 0.375, weighted-0.5 0.604). Weighted-0.5
  reaches 0.604 but at a control alarm rate of 24% vs PIRD's 8.8%, i.e. it
  buys detection with false positives, not with complementary attack signal.
* Pooled boundary (control-adjusted): PIRD α₅₀ = **1.17**; max α₅₀ is below the
  grid floor (already at global's noise level); weighted-0.5 α₉₀ = 1.78.

Only PIRD's detection is causally attributable to the per-ID manipulation.

---

## 9. Gradual-drift results (slow rate, control-adjusted)

`detection_significant` = fraction of configurations with a significant elevation
over the matched control (the naive "any alarm in 300 windows" metric is not used).

| α_end | PIRD | Global | max | OR | w0.5 |
|---|---|---|---|---|---|
| 0.25 | 0.062 | **0.00** | 0.042 | 0.042 | 0.062 |
| 0.50 | 0.188 | **0.00** | 0.062 | 0.062 | 0.125 |
| 0.75 | 0.229 | **0.00** | 0.104 | 0.104 | 0.146 |
| 1.00 | 0.271 | **0.00** | 0.125 | 0.125 | 0.167 |
| 1.50 | 0.417 | **0.00** | 0.208 | 0.292 | 0.208 |
| 2.00 | 0.583 | **0.00** | 0.333 | 0.438 | 0.396 |
| 3.00 | 0.792 | **0.00** | 0.417 | 0.562 | 0.521 |

Global never significantly detects gradual drift; hybrid rules are consistently
below PIRD's control-adjusted detection. PIRD is the most sensitive branch to
gradual legitimate-ID manipulation.

---

## 10. Conventional attack matrix

| Attack | PIRD | Global | Hybrid (max) |
|---|---|---|---|
| DoS (population flood) | ✗ recall 0.01 | ✓ recall 1.00 @ FPR 0.67 | ✓ recall 1.00 @ FPR 0.67 |
| Fuzzy (injection) | ✓ recall 0.98 @ FPR 0.14 | ~ recall 0.84 @ FPR 0.35 | ~ recall 0.95 @ FPR 0.34 |
| RPM (per-ID spoof) | ✓ recall 1.00 @ FPR 0.004 | ✗ recall 0.008 | ✓ recall 1.00 @ FPR 0.008 |
| gear (per-ID spoof) | ✓ recall 1.00 @ FPR 0.032 | ✗ recall 0.006 | ✓ recall 1.00 @ FPR 0.032 |
| Adaptive (per-ID) | partial, best branch | none | no gain over PIRD |
| Gradual (per-ID) | partial, best branch | none | no gain over PIRD |

The conceptual prediction — global handles population attacks, PIRD handles
per-ID attacks — holds. The practical problem is that global's false-positive
rate is unacceptable.

---

## 11. False-positive analysis

Pooled mean recall at fixed target FPRs (thresholds from val normals; **actual**
test FPR shown because the two diverge):

| Rule | target 0.1% | actual FPR | target 1% | actual FPR | target 5% | actual FPR |
|---|---|---|---|---|---|---|
| PIRD | recall 0.301 | 0.016 | 0.746 | 0.059 | 0.749 | 0.094 |
| Global | 0.456 | **0.239** | 0.464 | **0.257** | 0.488 | **0.290** |
| max | 0.555 | 0.170 | 0.987 | 0.261 | 0.999 | 0.295 |
| OR | 0.713 | 0.245 | 0.999 | 0.273 | 1.000 | 0.331 |
| w0.5 | 0.444 | 0.115 | 0.731 | 0.224 | 0.898 | 0.290 |
| w0.75 | 0.593 | 0.079 | 0.916 | 0.176 | 1.000 | 0.217 |

**The global branch cannot operate at low FPR**: even a 0.1% target yields ≈24%
actual test FPR, a train/test distribution shift (the test segment of each HCRL
file is a different bus regime). Consequently the hybrid's recall gains are
paid for with false alarms. At a genuinely comparable operating point (PIRD
FPR ≈ 1.6% → recall 0.30), the global-augmented systems need 7–24% FPR.
False alarms per hour are in `hybrid/conventional_metrics.csv`.

---

## 12. Complementarity

Detection overlap at the 1% branch thresholds (all test windows):

| Dataset | PIRD only | Global only | Both | Neither |
|---|---|---|---|---|
| DoS | 199 | **16882** | 841 | 5770 |
| Fuzzy | 493 | 4343 | 3570 | 12289 |
| RPM | **4971** | 159 | 39 | 18536 |
| gear | **5746** | 50 | 34 | 17872 |

Score correlation (test windows):

| Dataset | Pearson (all) | Spearman (all) | Pearson (normal) |
|---|---|---|---|
| DoS | 0.579 | 0.570 | 0.404 |
| Fuzzy | 0.523 | 0.497 | 0.587 |
| RPM | 0.102 | 0.080 | 0.116 |
| gear | −0.138 | −0.144 | −0.063 |

The branches are genuinely different detectors (near-zero/negative correlation
on RPM/gear). But on those datasets global's "unique" detections are almost
entirely false positives (global precision ≈ 0.01–0.19), so the complementary
information is not *useful* information. Complementary-but-useless.

---

## 13. Ablation (global feature families)

Adaptive attack, α=1.0, control-adjusted significant detection:

| Family | detection | control | significant |
|---|---|---|---|
| A bus | 0.261 | 0.271 | 0.00 |
| B ID-population | 0.204 | 0.213 | 0.00 |
| C cross-ID | 0.201 | 0.209 | 0.00 |
| D bus+ID-pop | 0.248 | 0.258 | 0.00 |
| E all | 0.243 | 0.253 | 0.00 |

No family detects the adaptive attack above its own control.

Conventional attacks (1% target; `hybrid/ablation_conventional.csv`):

| Dataset | family | recall | F1 | actual FPR |
|---|---|---|---|---|
| DoS | A bus | 0.999 | 0.491 | 0.676 |
| DoS | B ID-population | 1.000 | **0.497** | 0.568 |
| DoS | C cross-ID | 1.000 | 0.492 | 0.578 |
| DoS | D bus+ID-pop | 1.000 | 0.494 | 0.491 |
| DoS | E all | 1.000 | 0.495 | 0.666 |
| Fuzzy | A bus | 0.838 | 0.243 | 0.337 |
| Fuzzy | E all | 0.843 | 0.237 | 0.352 |
| RPM | A bus | 0.086 | 0.136 | 0.048 |
| RPM | E all | 0.008 | 0.015 | 0.009 |
| gear | A bus | 0.028 | 0.051 | 0.017 |
| gear | E all | 0.006 | 0.011 | 0.003 |

The bus (`A`) and ID-population (`B`) families drive population-attack detection;
the cross-ID family (`C`) adds little on its own. For the per-ID spoofing
attacks (RPM/gear) no family exceeds ~0.09 recall. On adaptive attacks all
families are at chance.

---

## 14. Limitations

* **Feature-space threat model.** The adaptive attack modifies per-ID window
  feature vectors; it does not correspond to a physically realizable frame
  modification. The global layer therefore sees only the induced aggregate
  value shifts. A raw-frame-level adaptive attacker could change timing/counts
  more directly and might be visible to the global layer; this is future work.
* **Train/test regime shift.** HCRL files have a strong distribution shift
  between the train and test segments; the global branch's calibrated threshold
  does not transfer, inflating test FPR. This is a property of the dataset
  protocol, reported honestly rather than corrected post hoc.
* **Window coupling.** Global features share the per-ID window table and its
  30-message smoothing; they are not computed from raw frames independently.
* **Single normalization family / detector.** Only robust and z-score
  normalization with Isolation Forest were evaluated.
* **No external validation run.** The ROAD raw `.log` captures are available
  locally, but the correct ROAD protocol is per-capture pre-injection
  calibration, which is a separate integration. The global feature/pipeline API
  is dataset-agnostic (`build_global_features` takes any window table); external
  validation is marked **pending**.
* **λ grid fixed.** The weighted grid was predefined and not tuned on attacks;
  no validation split was used to select a single λ.

---

## 15. Scientific conclusion

**The hypothesis is only partially supported, and not in the way that would
justify the hybrid for the adaptive threat.**

1. Global bus/population statistics **do** capture a genuinely different attack
   class: DoS flooding and (partly) Fuzzy injection are recovered where PIRD
   fails, and the two detectors are weakly/negatively correlated on per-ID
   spoofing data — so the representations are complementary in information.
2. However, the global branch **cannot detect the white-box adaptive
   legitimate-ID attacks at all** (control-adjusted significance 0.0 at every
   severity) and **does not detect gradual drift**. Adding it to PIRD does not
   recover PIRD's blind spot; it merely injects false alarms. At α ≤ 1.0 the
   hybrid's control-adjusted detection is equal to or worse than PIRD.
3. The global layer's recall gains on DoS/Fuzzy come at an **unacceptable
   false-positive cost** (≈24% actual FPR even at a 0.1% target). This is
   Outcome C (recall up, precision/deployment down) combined with Outcome B
   (little complementarity for the adaptive attack).

**Answer to the final research question:** a causal global/cross-ID
representation does provide *complementary information* (population attacks),
but **not for white-box adaptive legitimate-ID attacks**, and not while
maintaining an acceptable false-positive rate. PIRD remains the better single
branch for the per-ID threats studied here; the global layer's value is
confined to detection of population-level attacks at high FPR. A deployable
hybrid would require solving the global branch's calibration transfer problem
first, and would still not address adaptive per-ID evasion.
