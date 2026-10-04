# Phase 3 — Generalization & Transfer Plan

Status: implementation complete; dataset-limited execution.
Scope is fixed: **HCRL, ROAD, GEM-CAN, HCRL Survival Analysis**. No other
datasets are introduced.

---

## 1. Repository audit (Step 1)

| Concern | Location |
|---|---|
| PIRD per-ID residualization | `src/canguard/features/per_id.py` |
| Window/feature pipeline (14 features, ws=30) | `src/canguard/features/window.py`, `groups.py` |
| Temporal split | `src/canguard/features/splits.py` |
| Detectors / registry | `src/canguard/detectors/` |
| Threshold selection | `src/canguard/evaluation/threshold.py` |
| Metrics | `src/canguard/evaluation/metrics.py` |
| HCRL loader | `src/canguard/data/hcrl.py` |
| ROAD loader | `src/canguard/data/road.py` |
| ROAD pre-injection protocol | `src/canguard/exp/road_protocol.py`, `experiments/runners/run_phase_b_road.py` |
| Phase 2 frozen results | `experiments/phase2/`, `PHASE2_FINAL_REPORT.md` |

PIRD is reused unchanged. Phase 2 tests are preserved and must stay green.

---

## 2. Dataset audit (Step 2)

`python -m experiments.runners.phase3_dataset_audit --config experiments/configs/phase3_audit.yaml`

| Dataset | Available locally | Notes |
|---|---|---|
| HCRL Car-Hacking | **yes** (`data/{DoS,Fuzzy,RPM,gear}_dataset.csv`) | single vehicle (Hyundai YF Sonata); labels `R`/attack-type; no injection boundary metadata; 28 IDs |
| ROAD | **yes** (`road/road/{ambient,attacks}/*.log`) | per-capture injection intervals + injection IDs; 106 IDs/capture; independent captures |
| GEM-CAN | **yes** (`GEM_CAN_Dataset_R2/GEM_CAN_Dataset_R2/`) | GEM e6 autonomous EV; 2 sessions (normal-driving 100k, attack-scenario 43k); 29-bit extended IDs; DoS/Brake/Steering labels; 13–14 IDs |
| HCRL Survival Analysis | **yes** (`survival/dataset/`) | 3 vehicles (Sonata, Kia Soul, Spark); 4 captures each (free-driving + flooding/fuzzy/malfunction); 11-bit IDs; Flooding/Fuzzy/Malfunction labels |

Audit output: `experiments/phase3/dataset_audit/{dataset_manifest.csv,json,audit_summary.json,feature_compatibility.json}`.

**CAN-ID representation.** HCRL stores zero-padded hex (`0316`); ROAD stores bare
hex (`316`, `6e0`). A canonical ID (`int(hex) → lowercase hex, no leading zeros`)
is applied to **both** source and target before any statistics. This is a
representation normalisation, not a change to PIRD.

After normalisation, HCRL has **28** IDs and the evaluated ROAD captures have
**105**; the intersection is **3 IDs** (`0x130`, `0x153`, `0x430`). The 662-ID
figure sometimes quoted for ROAD is the per-capture count of a *fuzzing* capture,
whose injection introduces hundreds of random IDs; those are attack artifacts,
not shared normal IDs. Frame-weighted overlap is computed per transfer (2.9%
HCRL→ROAD, 11.3% ROAD→HCRL).

---

## 3. Frozen transfer protocol (Steps 3–4)

```
SOURCE DOMAIN
   source calibration normals  -> fit per-ID μ/σ
   source IF-fit normals       -> fit Isolation Forest
   source validation normals   -> choose 1% threshold
   FREEZE (stats, model, threshold)
        |
TARGET DOMAIN
   residualize with SOURCE stats (no target statistics)
   score with frozen model; compare to frozen threshold
```

The target never enters per-ID statistics, IF fitting, threshold selection,
feature/window selection, or preprocessing decisions. This is enforced by
construction (`fit_pird_source` receives only source frames) and by tests.

* **Primary result**: target windows whose canonical ID is present in the source
  per-ID statistics ("known-ID").
* **Diagnostic only**: target windows with unseen IDs, scored via the *existing*
  `transform_residuals` global fallback. Never mixed into the primary result.

Source normal splits:
* HCRL: per-file temporal 40/20/40; calib→stats, train(80%)→IF, train(20%)→threshold.
* ROAD: pooled pre-injection normals across eligible captures, deterministic
  60/20/20 (calib/IF/val) with `seed`.

---

## 4. Transfer matrix and status

GEM-CAN and HCRL Survival are now present (`GEM_CAN_Dataset_R2/`,
`survival/`), so their loaders and transfers are implemented and run.

| Source → Target | HCRL | ROAD | GEM-CAN | Survival |
|---|---|---|---|---|
| HCRL | — | **run** | **run** (0 shared IDs) | **run** (cross-vehicle) |
| ROAD | **run** | — | if valid (0 shared IDs) | if valid |
| GEM-CAN | **run** (0 shared IDs) | if valid (0 shared IDs) | — | if valid |
| Survival | **run** (cross-vehicle) | if valid | if valid | — |

Cross-capture within ROAD — **run**; within GEM-CAN (normal-driving vs
attack-scenario sessions) — **run**. Cross-vehicle: Sonata / Kia Soul /
Chevrolet Spark, all ordered pairs — **run** (runner mode `vehicle`).

**Important compatibility fact.** HCRL uses 11-bit IDs; GEM-CAN uses 29-bit
extended IDs. After canonicalisation the two datasets share **zero** IDs, so the
PIRD per-ID model has no source distribution for any GEM window. HCRL↔GEM-CAN is
therefore reported through the existing source-global fallback only, and the
primary known-ID metrics are undefined (NaN) by construction — not fabricated.

---

## 5. Metrics and schema

Primary: precision, recall, F1, ROC-AUC, PR-AUC, actual target FPR, source
validation FPR, source threshold, ID-count and frame-weighted overlap, unseen-ID
fractions. Diagnostic fixed-FPR: recall at source-selected 0.1 / 1 / 5 / 10%
(target actual FPR reported). Per-attack and per-target-capture breakdowns.

`n_source_frames` / `n_target_frames` in the result CSV are **PIRD window**
counts (one row per CAN-ID window), not raw CAN frame counts. Because PIRD
windows are per-ID, window-level and "frame-level" unseen fractions coincide.

Result schema: `docs/PHASE3_GENERALIZATION_REPORT.md` §5 and
`experiments/phase3/*/summary.csv`.

---

## 6. Implementation

```
src/canguard/data/gem_can.py                  # GEM-CAN loader (2 sessions)
src/canguard/data/survival.py                 # Survival multi-vehicle loader
src/canguard/evaluation/transfer.py           # source fit + zero-shot evaluation
src/canguard/evaluation/transfer_metrics.py   # ID overlap, unseen classification, recall@FPR
src/canguard/evaluation/transfer_shift.py      # source vs target SMD
experiments/runners/phase3_dataset_audit.py
experiments/runners/run_phase3_transfer.py     # modes: dataset, capture, gem_capture, vehicle
experiments/runners/phase3_shift.py            # consolidated SMD diagnostics
experiments/runners/phase3_figures.py          # transfer matrix + plots
experiments/configs/phase3_*.yaml
experiments/phase3/{dataset_audit,hcrl_to_road,road_to_hcrl,hcrl_to_gem,gem_to_hcrl,cross_capture,gem_captures,survival_vehicle,distribution_shift,results,plots}/
tests/test_phase3_transfer.py
tests/test_phase3_runner.py
tests/test_phase3_gem_survival.py
```

### 6.1 Loader / schema adaptations (documented, not PIRD changes)

* **GEM-CAN.** 29-bit extended IDs, hex payload bytes without zero padding,
  binary `label` + multi-class `Attack_Type`. Two independent sessions are
  exposed as captures (`normal_driving`, `attack_scenario`).
* **Survival.** Three vehicle directories (`Sonata`/`Soul`/`Spark` → canonical
  `sonata`/`kia_soul`/`chevrolet_spark`). Two payload encodings occur on disk and
  both are parsed: comma-separated bytes and a single space-separated payload
  field (KIA/Spark free-driving). Labels `R`/`T`; free-driving captures are
  unlabeled normal.
* `FeaturePipeline` now carries `attack_type`/`capture` through to the window
  table when those columns are present (additive; behavioural features
  unchanged).

### 6.1 Compatibility / plumbing fixes (not PIRD changes)

* `_hcrl_window_table` now takes the explicit source/target section. Previously
  it always read `data_dir`/`sample_size` from `cfg["source"]`, so the
  ROAD→HCRL target loaded the full HCRL file (`sample_size=None`) and ran for
  30+ minutes; with the fix it completes in seconds.
* Figure runner: the cross-capture heatmap no longer creates duplicate
  `source_dataset`/`target_dataset` columns.
* `validate_domain_schema`, `split_captures`, `split_vehicles` and
  `vehicle_domains` were added to `src/canguard/evaluation/transfer.py` for
  explicit schema checks and leakage-free capture/vehicle partitioning.

ROAD-specific pre-injection calibration remains Phase 2 external validation and
is **not** cross-dataset transfer.

---

## 7. GEM-CAN / Survival interface (pending)

When the data is placed under `data/GEM-CAN` and `data/Survival`:

1. Add a loader producing the canonical schema
   (`timestamp, can_id, dlc, data_0..7, label, is_attack`) plus `attack_type`.
2. For Survival, attach a `vehicle` column and expose per-vehicle captures.
3. Register the loader in `canguard.data.factory` and add dataset builders in
   `run_phase3_transfer._build_source/_build_target` (mirroring HCRL/ROAD).
4. Do **not** retune any PIRD parameter; reuse the frozen configs.

No target-normal data may be used for calibration, threshold selection, model
fitting, feature/window selection, or ID fallback design.
