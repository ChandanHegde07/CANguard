# Phase 3 — Generalization & Transfer Report

Frozen PIRD. Four dataset families: **HCRL**, **ROAD**, **GEM-CAN**, **HCRL
Survival**. This report keeps three protocols strictly separate:

* **External validation** — ROAD calibration → ROAD test (Phase 2; already done).
* **Cross-dataset transfer** — e.g. HCRL calibration → GEM-CAN test (zero-shot).
* **Cross-vehicle transfer** — Vehicle A calibration → Vehicle B test (zero-shot).

No target normal traffic was used for per-ID fitting, Isolation Forest training,
threshold selection, feature/window selection, or preprocessing anywhere below.
Target-recalibration numbers are explicitly labelled diagnostic and are never
presented as generalization.

Machine-readable results: `experiments/phase3/*/summary.csv`,
`survival_vehicle/vehicle_transfer.csv`, `gem_captures/capture_transfer.csv`,
`cross_capture/capture_transfer.csv`, `distribution_shift/*`, `results/*`.

---

## 1. Executive Summary

1. **Zero-shot cross-dataset transfer fails, in every tested direction.** At the
   source-selected 1% threshold the target FPR is 1.00 (HCRL↔ROAD, zero-overlap
   HCRL↔GEM-CAN fallback, and most cross-vehicle directions). Recall is 1.00 only
   trivially ("flag everything"); otherwise F1 collapses to ≈ the positive base
   rate.
2. **HCRL and GEM-CAN share zero CAN IDs.** HCRL uses 11-bit IDs, GEM-CAN uses
   29-bit extended IDs; after canonicalisation the intersection is empty. PIRD's
   per-ID model therefore has no source distribution for *any* GEM-CAN window.
   The known-ID result is undefined (NaN) by construction and only the existing
   source-global fallback can score these windows — reported as a limitation, not
   patched.
3. **Cross-vehicle transfer is ID-overlap dependent.** Hyundai Sonata ↔ Kia Soul
   share 22 IDs and transfer partially (recall 0.94–0.98, F1 0.15–0.34, but FPR
   0.25–0.47). Chevrolet Spark shares only 2–3 IDs with either and collapses to
   FPR 1.0.
4. **The dominant failure mechanism is normal-regime / calibration shift
   (Case A), not attack-signature loss.** Target recalibration — a diagnostic —
   restores in-domain F1 to 0.80–0.88 for the Survival vehicles, 0.66–0.88 for
   HCRL/ROAD, but only 0.15 for GEM-CAN (whose attacks are intrinsically hard for
   this 14-feature pipeline).
5. **Cross-capture within a vehicle is partial and regime-dependent**: ROAD 3/30
   pairs at F1 ≥ 0.5 and FPR ≤ 0.05; GEM-CAN's two sessions fail (F1 0.10 / 0.0).
6. **No target leaks.** `fit_pird_source` receives only source frames; the target
   is passed solely to `evaluate_transfer`. Deterministic seeds and source-only
   thresholds are unit-tested.

---

## 2. Research Questions

| RQ | Question | Answer |
|---|---|---|
| RQ1 | PIRD transfer HCRL → GEM-CAN? | **No per-ID transfer possible** (0 shared IDs); fallback FPR 1.0, F1 0.458. |
| RQ2 | PIRD transfer GEM-CAN → HCRL? | **No** (0 shared IDs); fallback flags nothing (FPR 0.0). |
| RQ3 | PIRD transfer across Survival vehicles? | **Partially**, only for Sonata↔Kia Soul (22 shared IDs); Spark collapses. |
| RQ4 | Is transfer symmetric? | Magnitude is asymmetric (e.g. Kia→Sonata F1 0.342 vs Sonata→Kia 0.153); outcomes are symmetric failures where IDs are disjoint. |
| RQ5 | How much CAN-ID overlap exists? | HCRL∩ROAD = 3; HCRL∩GEM = 0; Sonata∩Kia = 22; Spark∩{Sonata,Kia} = 2–3. |
| RQ6 | Does normal shift explain FPR degradation? | **Yes** — high target FPR is the primary failure (Case A); SMD tracks it. |
| RQ7 | Does attack shift explain recall degradation? | Only where FPR stays controlled; generally recall is preserved (degenerately) when FPR → 1. |
| RQ8 | How much does recalibration recover? | In-domain F1 0.66–0.88 (HCRL/ROAD/Survival); 0.15 (GEM-CAN). |
| RQ9 | Does PIRD learn vehicle-independent structure? | **No** — per-ID statistics are environment-conditioned; only shared-ID, similar-regime pairs transfer partially. |
| RQ10 | Operating envelope of PIRD? | Same bus/vehicle, sufficient ID overlap and matched normal regime, at the cost of FPR. |

---

## 3. Dataset Audit

Run: `python -m experiments.runners.phase3_dataset_audit --config experiments/configs/phase3_audit.yaml`

Output: `experiments/phase3/dataset_audit/`. **available = {hcrl, road, gem_can,
hcrl_survival}; missing = {}; survival vehicles = {sonata, kia_soul,
chevrolet_spark}.**

| Dataset | Structure | IDs | Labels | Boundary known | Captures independent | Vehicle |
|---|---|---|---|---|---|---|
| HCRL | 4 CSVs (`DoS,Fuzzy,RPM,gear`) | 28 (11-bit) | `R` / attack-type | no | no (temporal) | Hyundai YF Sonata |
| ROAD | 45 logs (12 ambient, 33 attacks) | 106/capture (662 fuzzing) | interval+injection-id | yes | yes | 2010 Toyota Highlander |
| GEM-CAN | 2 CSVs: normal-driving (100,000 frames), attack-scenario (43,444) | 13 / 14 (29-bit extended) | 0/1 + `Attack_Type` (DoS, Brake/Steering Tampering) | yes (`attack_scenario`) | yes (two sessions) | GEM e6 autonomous EV |
| HCRL Survival | 3 vehicle dirs × 4 captures | 27–83 normal (up to 2048 with fuzzy-injected IDs) | `R` / `T` | no | yes (separate sessions) | Sonata, Kia Soul, Spark |

**GEM-CAN.** Payload bytes are hex without zero padding (`5` = 0x05); the attack
session contains 42,405 attack + 1,039 normal frames. The DoS attack floods ID
`0x0` (unseen in normal traffic), and Brake/Steering tampering use legitimate
extended IDs (`18FF00F9`, `18FF70F9`) with altered payloads.

**HCRL Survival.** Vehicle directories map to `sonata` (`*_SONATA`), `kia_soul`
(`*_KIA`), `chevrolet_spark` (`*_Spark`). Two payload encodings occur on disk and
both are parsed: comma-separated bytes, and a single space-separated payload
field (KIA/Spark free-driving captures). Labels are `R`/`T`; free-driving
captures are unlabeled normal. Attack families: flooding, fuzzy, malfunction.

**CAN-ID representation.** HCRL/ROAD/Survival use 11-bit IDs; GEM-CAN uses 29-bit
extended IDs. A canonical form (`int(hex)` → lowercase hex, no leading zeros) is
applied to both domains before statistics. This is representation normalisation
only and changes no PIRD parameter.

---

## 4. Experimental Protocol

```
SOURCE DOMAIN (only source data is touched)
    source calibration normals  -> per-ID mu/sigma        (fit_per_id_stats)
    source IF-fit normals       -> Isolation Forest        (200 trees, seed 0)
    source validation normals   -> 1% threshold (99th pct)
    FREEZE
        |
TARGET DOMAIN (evaluation only)
    residualize with SOURCE stats; score with frozen model; frozen threshold
        |
    primary result  = target windows whose ID has a source per-ID distribution
    diagnostic only = unseen-ID windows via the existing global fallback
```

* HCRL source normals: per-file temporal 40/20/40 (calib→stats, train 80%→IF,
  20%→threshold).
* ROAD source normals: pooled pre-injection normals, deterministic 60/20/20.
* GEM-CAN source normals: normals pooled from both sessions, deterministic
  60/20/20.
* Survival source normals: per-vehicle normals pooled across that vehicle's four
  captures, deterministic 60/20/20.
* Detector/window/feature settings are unchanged from Phase 2. No target tuning.

**Unit of accounting.** PIRD windows are per CAN ID, so each window belongs to
exactly one ID; window-level and "frame-level" unseen fractions necessarily
coincide. `n_source_frames`/`n_target_frames` in the result CSVs are PIRD **window**
counts.

---

## 5. HCRL → ROAD (cross-dataset, zero-shot)

| Metric | Value |
|---|---|
| Source / target windows | 112,596 / 338,165 |
| Source / target / shared IDs | 25 / 105 / **3** |
| Frame-weighted overlap | 0.029 |
| Unseen-ID window fraction | 0.971 |
| Source threshold / validation FPR | 0.506 / 0.0101 |
| **Actual target FPR (known-ID)** | **1.000** |
| Recall / precision / F1 | 1.000 / 0.033 / 0.064 |
| ROC-AUC / PR-AUC | 0.634 / 0.137 |
| Mean / median / max abs SMD | 0.733 / 0.820 / 1.513 |

No source-selected threshold controls target FPR (0.1–10% all give FPR 1.0).
Most-shifted features: `byte_var` (−1.51), `byte_max_change` (−1.31),
`byte_mean` (−1.16), `iat_min` (+1.12).

## 6. ROAD → HCRL (cross-dataset, zero-shot)

| Metric | Value |
|---|---|
| Source / target windows | 250,426 / 91,794 |
| Source / target / shared IDs | 77 / 28 / **3** |
| Frame-weighted overlap | 0.113 |
| Unseen-ID window fraction | 0.887 |
| Source threshold / validation FPR | 0.576 / 0.0100 |
| **Actual target FPR (known-ID)** | **1.000** |
| Recall / precision / F1 | 1.000 / 0.017 / 0.033 |
| ROC-AUC / PR-AUC | 0.592 / 0.103 |
| Mean / median / max abs SMD | 0.271 / 0.264 / 0.897 |

Asymmetric magnitude (HCRL→ROAD shift 0.73 vs 0.27) but symmetric failure.

---

## 7. GEM-CAN

### 7.1 Dataset audit

See §3. GEM-CAN has **no CAN IDs in common with HCRL or ROAD** (11-bit vs 29-bit
extended). This is a hard domain incompatibility for an ID-conditioned model, not
a tunable defect.

### 7.2 HCRL → GEM-CAN

Source: HCRL normals. Target: GEM-CAN normal-driving + attack-scenario windows.

| Metric | Value |
|---|---|
| Source / target windows | 112,596 / 142,670 |
| Source / target / shared IDs | 25 / 14 / **0** |
| Frame-weighted overlap | 0.000 |
| Unseen-ID window fraction | **1.000** |
| Source threshold / validation FPR | 0.506 / 0.0101 |
| Primary (known-ID) metrics | **undefined (NaN) — no shared IDs** |
| Fallback precision / recall / F1 | 0.297 / 1.000 / 0.458 |
| Fallback actual FPR | **1.000** |
| Mean / median / max abs SMD | 0.563 / 0.384 / 1.301 |

The global fallback flags every window, so its recall is degenerate. This is
**Outcome D/E**: no per-ID transfer is possible, and the only available scorer
collapses.

### 7.3 GEM-CAN → HCRL

| Metric | Value |
|---|---|
| Source / target windows | 100,352 / 91,794 |
| Source / target / shared IDs | 13 / 28 / **0** |
| Unseen-ID window fraction | **1.000** |
| Source threshold / validation FPR | 0.586 / 0.0100 |
| Primary (known-ID) metrics | **undefined (NaN)** |
| Fallback precision / recall / F1 | 0.000 / 0.000 / 0.000 |
| Fallback actual FPR | 0.000 |
| Mean / median / max abs SMD | 0.563 / 0.391 / 1.301 |

The fallback flags nothing here — the opposite degenerate extreme. Both
directions confirm that ID-disjoint transfer is meaningless for PIRD as specified.

### 7.4 GEM-CAN capture transfer

GEM-CAN has two genuinely independent sessions. Both directions were run
(`run_gem_capture_transfer`):

| Source capture | Target capture | shared/target IDs | overlap | FPR | recall | F1 |
|---|---|---|---|---|---|---|
| normal_driving | attack_scenario | 12 / 13 | 0.046 | 0.586 | 0.069 | 0.098 |
| attack_scenario | normal_driving | 10 / 13 | 0.364 | 0.183 | 0.000 | 0.000 |

Even within the same vehicle, cross-session transfer is poor: the target FPR is
inflated and recall is low. The low overlap for normal→attack (0.046) is because
the DoS flood uses ID `0x0`, which never appears in normal driving.

---

## 8. Cross-Capture Transfer (ROAD)

Artifacts: `experiments/phase3/cross_capture/capture_transfer.csv` (30 pairs).

| Statistic | F1 | Recall | Precision | ROC-AUC | Actual FPR |
|---|---|---|---|---|---|
| mean | 0.310 | 0.845 | 0.233 | 0.896 | 0.237 |
| median | 0.285 | 0.998 | 0.179 | 0.959 | 0.252 |
| min / max | 0.000 / 0.802 | 0.000 / 1.000 | 0.000 / 0.691 | 0.494 / 1.000 | 0.022 / 0.475 |

3/30 pairs reach F1 ≥ 0.5 at FPR ≤ 0.05 (best: `max_speedometer_attack_1` →
`reverse_light_off_attack_1`, F1 0.802 at FPR 0.022). Frame-weighted ID overlap is
high (mean 0.966), so degradation is regime-driven, not ID-driven.

---

## 9. Survival Analysis

### 9.1 Dataset audit

Three vehicles, four captures each, verified from disk (see §3). Per-vehicle
normal-traffic ID counts differ sharply: Sonata 27, Kia Soul 45, Chevrolet Spark
81; the fuzzy captures inject up to 2,048 random IDs.

### 9.2 Vehicle-level transfer

Artifacts: `experiments/phase3/survival_vehicle/vehicle_transfer.csv` (6 ordered
pairs) and per-pair subdirectories.

| Source → Target | shared/target IDs | frame overlap | unseen frac | FPR | recall | F1 | ROC-AUC | mean \|SMD\| |
|---|---|---|---|---|---|---|---|---|
| Sonata → Kia Soul | 22 / 80 | 0.621 | 0.379 | 0.473 | 0.976 | 0.153 | 0.933 | 0.153 |
| Sonata → Chevrolet Spark | 3 / 83 | 0.004 | 0.996 | 1.000 | 1.000 | 0.043 | 0.956 | 0.692 |
| Kia Soul → Sonata | 22 / 28 | 0.728 | 0.272 | 0.253 | 0.942 | 0.342 | 0.917 | 0.219 |
| Kia Soul → Chevrolet Spark | 2 / 83 | 0.004 | 0.996 | 1.000 | 1.000 | 0.045 | 0.997 | 0.829 |
| Chevrolet Spark → Sonata | 3 / 28 | 0.116 | 0.884 | 1.000 | 1.000 | 0.053 | 0.711 | 0.540 |
| Chevrolet Spark → Kia Soul | 3 / 80 | 0.027 | 0.973 | 1.000 | 1.000 | 0.082 | 0.997 | 0.603 |

Per-attack (known-ID windows): for Sonata→Kia Soul, `fuzzy` F1 0.964 and
`malfunction` F1 1.000 at FPR 0; for Kia Soul→Sonata, `fuzzy` 0.871 and
`malfunction` 0.999. The overall F1 is dragged down by the normal-window FPR, not
by missed attacks — i.e. **Case A**.

**Interpretation.** Transfer is only meaningful between vehicles that share IDs
and normal regime (Sonata ↔ Kia Soul, both 11-bit, same platform group): recall is
preserved (0.94–0.98) but FPR rises (0.25–0.47), so F1 is modest. Chevrolet Spark
shares almost no IDs with the others, and all Spark-related directions collapse
to FPR 1.0 — again dominated by unseen IDs and normal shift, not attack shift.

### 9.3 Per-vehicle (in-domain) recalibration diagnostic

| Recalibrated on | F1 | recall | precision | FPR |
|---|---|---|---|---|
| Sonata | 0.883 | 0.923 | 0.846 | 0.010 |
| Kia Soul | 0.800 | 0.836 | 0.766 | 0.010 |
| Chevrolet Spark | 0.874 | 0.982 | 0.788 | 0.010 |

In-domain PIRD is strong; cross-vehicle degradation is largely recoverable with
target normal traffic — confirming calibration dependence.

---

## 10. Known vs Unseen CAN IDs

| Transfer | shared/target | frame overlap | unseen window fraction | primary result |
|---|---|---|---|---|
| HCRL → ROAD | 3 / 105 | 0.029 | 0.971 | NaN (fallback) |
| ROAD → HCRL | 3 / 28 | 0.113 | 0.887 | NaN (fallback) |
| HCRL → GEM-CAN | 0 / 14 | 0.000 | 1.000 | undefined |
| GEM-CAN → HCRL | 0 / 28 | 0.000 | 1.000 | undefined |
| Sonata → Kia Soul | 22 / 80 | 0.621 | 0.379 | defined |
| Kia Soul → Sonata | 22 / 28 | 0.728 | 0.272 | defined |
| Spark ↔ {Sonata, Kia} | 2–3 / 28–83 | ≤0.12 | ≥0.88 | degenerate |

Unseen IDs are never silently dropped; they are counted and scored only by the
existing source-global fallback, reported separately.

---

## 11. Distribution Shift

Artifacts: `experiments/phase3/distribution_shift/`. SMD is a diagnostic effect
size, not causal evidence. Feature-level SMD uses target KNOWN-ID normals when
available, otherwise all target normals (zero-overlap cases).

| Transfer | mean \|SMD\| | median \|SMD\| | max \|SMD\| | most shifted |
|---|---|---|---|---|
| HCRL → ROAD | 0.733 | 0.820 | 1.513 | byte_var |
| ROAD → HCRL | 0.271 | 0.264 | 0.897 | byte_var |
| HCRL → GEM-CAN | 0.563 | 0.384 | 1.301 | iat_min |
| GEM-CAN → HCRL | 0.563 | 0.391 | 1.301 | iat_min |
| Sonata → Kia Soul | 0.153 | 0.073 | 0.505 | byte_mean |
| Sonata → Spark | 0.692 | 0.903 | 1.318 | — |
| Kia Soul → Sonata | 0.219 | 0.147 | 0.724 | — |
| Kia Soul → Spark | 0.829 | 0.931 | 1.730 | — |

Payload statistics (`byte_*`) and timing statistics (`iat_*`) dominate. Per-ID SMD
(`all_per_id_shift.csv`) is far larger for several shared IDs (e.g. Sonata→Kia
`2c0` mean |SMD| 105, Kia→Sonata `370` 430), driven by near-zero-variance
features — evidence that "shared" IDs do not share a behavioural distribution.
The larger feature-level SMD for Spark directions (0.69–0.83) aligns with their
FPR = 1.0 collapse.

---

## 12. Failure Analysis

* **HCRL ↔ ROAD and HCRL/GEM-CAN dataset transfer — Case A (extreme).** Target
  normal traffic alone produces FPR = 1.0 at every source threshold, so attack
  recall is unidentifiable at a controlled operating point. For zero-overlap
  GEM-CAN, per-ID scoring is impossible.
* **Cross-vehicle Sonata ↔ Kia Soul — Case A.** Controlled-ish FPR is not
  achieved (0.25–0.47) while recall stays high (0.94–0.98); named attack types on
  shared IDs are detected at FPR 0.
* **Cross-vehicle Spark directions — Case A (extreme).** FPR = 1.0, recall 1.0
  degenerate; driven by unseen IDs.
* **GEM-CAN cross-capture — Case C.** High FPR (0.59) and low recall (0.07):
  neither normal regime nor attack signature transfers between GEM sessions.
* **ROAD cross-capture — Case A with partial success.** 3/30 pairs match regime
  and transfer well; the rest inflate FPR.

No direction exhibits controlled FPR with collapsed recall in isolation (Case B),
because whenever FPR is controlled the recall is also preserved.

---

## 13. Target-Recalibration Diagnostic

**Not zero-shot.** Fits PIRD on the target domain's own normals and evaluates the
same target test.

| Diagnostic | F1 | recall | precision | FPR |
|---|---|---|---|---|
| ROAD normals → ROAD test | 0.656 | 0.638 | 0.675 | 0.024 |
| HCRL normals → HCRL test | 0.879 | 0.978 | 0.798 | 0.038 |
| GEM-CAN normals → GEM-CAN test | 0.151 | 0.143 | 0.159 | 0.010 |
| Sonata normals → Sonata test | 0.883 | 0.923 | 0.846 | 0.010 |
| Kia Soul normals → Kia Soul test | 0.800 | 0.836 | 0.766 | 0.010 |
| Chevrolet Spark normals → Spark test | 0.874 | 0.982 | 0.788 | 0.010 |

Recalibration recovers most of the HCRL/ROAD/Survival degradation, confirming the
zero-shot failures are dominated by source-calibration mismatch. GEM-CAN remains
weak even with recalibration — its attacks (payload-only tampering on legitimate
IDs, and a DoS flood on an unseen ID) are intrinsically hard for this 14-feature
per-ID pipeline.

---

## 14. Results

Consolidated: `experiments/phase3/results/{transfer_matrix.csv,
capture_transfer.csv, vehicle_transfer.csv, phase3_summary.json}` and
`distribution_shift/shift_summary.csv`.

### 14.1 Final transfer matrix (dataset level — executed cells only)

| Source → Target | HCRL | ROAD | GEM-CAN | Survival |
|---|---|---|---|---|
| HCRL | — | ✓ | ✓ (0 shared) | — |
| ROAD | ✓ | — | not run (0 shared) | — |
| GEM-CAN | ✓ (0 shared) | not run (0 shared) | — | — |
| Survival | — | — | — | — |

`✓` = executed. Dataset-level transfers into/out of Survival were **not** run;
Survival is evaluated through the cross-vehicle matrix below. "0 shared" marks
executed transfers whose primary known-ID metrics are undefined.

### 14.2 Cross-vehicle matrix (executed)

| Source vehicle → Target vehicle | Sonata | Kia Soul | Spark |
|---|---|---|---|
| Sonata | — | ✓ F1 0.153 / FPR 0.473 | ✓ F1 0.043 / FPR 1.000 |
| Kia Soul | ✓ F1 0.342 / FPR 0.253 | — | ✓ F1 0.045 / FPR 1.000 |
| Spark | ✓ F1 0.053 / FPR 1.000 | ✓ F1 0.082 / FPR 1.000 | — |

### 14.3 Cross-capture matrices

* ROAD: 30 ordered pairs, mean F1 0.310, mean FPR 0.237 (see §8).
* GEM-CAN: 2 ordered pairs, F1 0.098 and 0.000 (see §7.4).

---

## 15. Visualization

Generated under `experiments/phase3/plots/`:

* `transfer_f1.png`, `transfer_recall.png`, `transfer_fpr.png` — dataset-level
  transfer heatmaps (HCRL / ROAD / GEM-CAN).
* `vehicle_transfer_f1.png`, `vehicle_transfer_recall.png`,
  `vehicle_transfer_fpr.png`, `vehicle_id_overlap.png` — cross-vehicle matrix.
* `capture_transfer_f1.png`, `gem_capture_transfer_f1.png` — cross-capture.
* `smd_*.png`, `per_id_smd.png`, `shift_means.png` — distribution shift.
* `id_overlap.png`, `transfer_f1_bars.png`, `operating_*.png` — ID overlap,
  known-vs-unseen F1, source-selected operating curves.

---

## 16. Limitations

* HCRL and ROAD are different vehicles and datasets, so dataset and vehicle
  identity are confounded for HCRL↔ROAD.
* HCRL↔GEM-CAN is a pure unseen-ID regime (0 shared IDs); no per-ID claim can be
  made.
* Cross-vehicle result is limited to three vehicles from one acquisition campaign;
  Spark's tiny ID overlap makes its directions degenerate.
* Cross-dataset attack-shift analysis is unidentifiable where normal shift is
  total (FPR = 1.0).
* GEM-CAN's attack session has only 1,039 normal frames; its normal-driving
  session is a separate regime.
* Unseen-ID evaluation uses PIRD's existing source-global fallback only; no new
  fallback was invented.
* Per-ID windows make frame- and window-level unseen fractions coincide.
* No target tuning of any kind; recalibrated numbers are labelled diagnostic.

---

## 17. Reproducibility

```
python -m experiments.runners.phase3_dataset_audit   --config experiments/configs/phase3_audit.yaml
python -m experiments.runners.run_phase3_transfer    --config experiments/configs/phase3_hcrl_to_road.yaml --mode dataset
python -m experiments.runners.run_phase3_transfer    --config experiments/configs/phase3_road_to_hcrl.yaml --mode dataset
python -m experiments.runners.run_phase3_transfer    --config experiments/configs/phase3_hcrl_to_gem.yaml  --mode dataset
python -m experiments.runners.run_phase3_transfer    --config experiments/configs/phase3_gem_to_hcrl.yaml  --mode dataset
python -m experiments.runners.run_phase3_transfer    --config experiments/configs/phase3_capture_transfer.yaml --mode capture
python -m experiments.runners.run_phase3_transfer    --config experiments/configs/phase3_gem_capture_transfer.yaml --mode gem_capture
python -m experiments.runners.run_phase3_transfer    --config experiments/configs/phase3_vehicle_transfer.yaml --mode vehicle
python -m experiments.runners.phase3_shift   --config experiments/configs/phase3_shift.yaml
python -m experiments.runners.phase3_figures --config experiments/configs/phase3_figures.yaml
pytest
```

**Loader / schema adaptations (no PIRD changes):** new `GemCanLoader` and
`SurvivalLoader` produce the canonical frame schema; `FeaturePipeline` now carries
`attack_type`/`capture` through to the window table when present (additive).
CAN-ID canonicalisation is representation-only. An earlier HCRL target-config bug
is fixed. All PIRD settings (14 features, `window_size=30`, Isolation Forest
200/seed 0, 1% source-validation threshold) are unchanged.

---

## 18. Updated Generalization Summary

### Supported findings

1. PIRD does **not** zero-shot transfer across HCRL ↔ ROAD, HCRL/GEM-CAN, or
   GEM-CAN's own two sessions: FPR inflates to 1.0 or recall degenerates.
2. The dominant failure is normal-regime / source-calibration shift (Case A);
   target recalibration restores in-domain F1 to 0.66–0.88.
3. CAN-ID overlap governs feasibility: 0 shared IDs (HCRL/GEM) → undefined;
   2–3 (Spark) → collapse; 22 (Sonata/Kia) → partial transfer.
4. Cross-vehicle Sonata ↔ Kia Soul transfers with high recall but inflated FPR;
   cross-capture ROAD transfers in a regime-dependent subset (3/30).

### Partially supported

* "PIRD reuses behavioural structure across environments." Supported only for
  same-vehicle/same-regime captures and high-ID-overlap vehicle pairs; not a
  general property.
* "Transfer is symmetric." Magnitude is asymmetric; failures are symmetric when
  IDs are disjoint.

### Unsupported hypotheses

* PIRD generalizes across datasets or vehicles without recalibration.
* PIRD's per-ID representation is environment-invariant.
* The source-global fallback rescues unseen-ID windows (it does not: FPR 0–1
  degenerate, F1 ≤ 0.46).

### Remaining open questions

* Whether any PIRD-adjacent representation (not a redesign of Phase 3) could
  bridge the 29-bit GEM-CAN ID space.
* Data-efficiency of target recalibration (how few target normals suffice).
* Cross-vehicle transfer with more vehicles/platforms than the three available.
