# Phase 2 Plan — White-Box Adaptive Attacks Against PIRD

**Status:** planning + first-pass implementation.
**Scope of this pass:** establish the frozen PIRD baseline, implement a white-box
adaptive attacker (fixed severity + gradual drift), measure the detection
boundary, and compare adaptive vs non-adaptive attacks. **PIRD itself is not
modified.**

---

## 1. Relevant existing files

| Concern | File | Notes |
|---|---|---|
| Dataset loaders | `src/canguard/data/{base,hcrl,road,factory}.py` | HCRL CSV → canonical schema (`timestamp,can_id,dlc,data_0..7,label`) + `is_attack`. |
| Frame→window features | `src/canguard/features/window.py` | `PerIDWindow`, `FeaturePipeline`, `fit_known_ids_on_normal_prefix`. Sliding window per ID; emits a feature vector for **every** message once the window is full (ws=30). |
| Feature names | `src/canguard/features/groups.py` | `BEHAVIORAL_FEATURES_V1` (14 features), groups IAT/byte/DLC/other. |
| Per-ID residualization (PIRD core) | `src/canguard/features/per_id.py` | `fit_per_id_stats` (μ/σ per ID from **calib normals only**), `transform_residuals` (`r=(x-μ)/(σ+EPS)`), global fallback, `MIN_WINDOWS_PER_ID=20`, `EPS=1e-6`. |
| Temporal split | `src/canguard/features/splits.py` | `temporal_split(ft, 0.4, 0.2, 0.4)`. |
| Detectors | `src/canguard/detectors/` | `IsolationForestDetector` + registry. Score convention: higher = more anomalous. |
| Train + threshold + eval | `src/canguard/evaluation/evaluation.py` | `train_anomaly_detector`: fit on train **normals**, threshold on last 20% of train normals (`choose_threshold_from_val_normals`, percentile at `fpr_target=0.01`), score test. |
| Metrics | `src/canguard/evaluation/metrics.py` | precision/recall/F1/FPR/ROC-AUC/PR-AUC, tp/fp/fn/tn. |
| Latency | `src/canguard/evaluation/latency.py` | `detection_latency`, `multi_segment_latencies`. |
| Bootstrap CIs | `src/canguard/evaluation/bootstrap.py` | block bootstrap. |
| Experiment infra | `src/canguard/exp/` | `config`, `tracking` (`ExperimentRun`, `save_json/csv`), `seeds`, `logging_utils`, `cache` (`FeatureCache`), `matrix` (`build_window_table`, `build_raw_and_residual_splits`, `resolve_data_path`). |
| Runners / configs | `experiments/runners/`, `experiments/configs/` | `train_detector`, `run_phase_a`, ROAD runners; `hcrl.yaml`, `phase_a.yaml`, … |
| Plotting | `src/canguard/visualization/` | `curves`, `distributions`, `timeline`, `matrices`, `style.apply_ieee_style`, `save_ieee_figure`. |
| Tests | `tests/` | pytest suite; `test_experiments_runner.py` builds a tiny synthetic HCRL CSV. |

**Data note.** The HCRL CSVs live in `data/` (`DoS_dataset.csv`, `Fuzzy_dataset.csv`,
`RPM_dataset.csv`, `gear_dataset.csv`), not `HCRL Car-Hacking/`. `hcrl.yaml` and
`run_pipeline.py` point at the empty `HCRL Car-Hacking/` dir; `resolve_data_path`
falls back to `data/`. Phase 2 configs use `data_dir: data` explicitly. This is an
environment/path fix, **not** a methodology change.

---

## 2. How PIRD currently works (exact pipeline)

```
raw CAN frames
  └─ FeaturePipeline(window_size=30, known_ids=IDs seen in first 20k normal rows)
       → per-ID sliding-window feature vector (14 BEHAVIORAL_FEATURES_V1)
  └─ temporal_split(ft, 0.4/0.2/0.4) = (calib, train, test)
  └─ fit_per_id_stats(calib normals) → per-ID μ_k, σ_k (+ global fallback)
  └─ transform_residuals(x; μ_k, σ_k) → r = (x − μ_k)/(σ_k + EPS)
  └─ IsolationForest(n_estimators=200, random_state=0) fit on train **normals**
  └─ threshold = (1−0.01) percentile of scores on last 20% of train normals
  └─ predict on test: score ≥ threshold
```

* **Normalization statistics are calculated in** `fit_per_id_stats`, called on
  `calib[is_attack == 0]` only.
* **Anomaly scores are generated in** `IsolationForestDetector.score_samples`
  (`-sklearn.score_samples`), applied to residual columns `*_res`.
* **Attacks are represented** only as dataset labels (`label != "R"` →
  `is_attack=1`). There is **no attack generator** in the repository. The
  “existing/non-adaptive attack” is therefore the real HCRL injection present in
  the test split.
* **Experiments are executed** via `python -m experiments.runners.<name> --config <yaml>`;
  results are written under `experiments/<phase>/runs/<UTC-stamp>/` plus
  consolidated `tables/` / `figures/`.

---

## 3. Phase 2 experimental design

### 3.1 Threat model (white-box)

The attacker knows: CAN IDs, the 14-feature PIRD representation, the frozen
per-ID normal means/stds `(μ_k, σ_k)`, the normalization `r=(x−μ)/(σ+EPS)`, the
Isolation Forest and the frozen threshold. The attacker compromises an
**existing legitimate ID** and replaces its window feature vector with

```
x_attack = μ_k + α · σ_k · direction          (per targeted feature)
```

then clips to physically valid (bounded/discrete) feature ranges. Because the
residual transform is linear in x, this produces residual `≈ α · direction`
before clipping. **No statistic is ever estimated from attack/test data** —
everything comes from calib/train.

Constraint handling: per-feature bounds (`byte_mean∈[0,255]`,
`byte_entropy∈[0,8]`, `byte_nunique∈[0,240]`, `dlc_mode∈{0..8}`, IAT/other ≥ 0,
`window_fill∈[0,1]`), with integer rounding for `dlc_mode`/`byte_nunique`.
Negative directions on non-negative features saturate — reported honestly.

### 3.2 Injection substrate (no test-stat leakage)

* Calib → fit `(μ_k, σ_k)`. Train normals → fit IF + frozen threshold.
* Test **normal** windows (`is_attack == 0`, ~18k/dataset) form the substrate.
  Their residuals/scores are precomputed once.
* For a target legitimate ID `k`, a contiguous block of its test-normal windows
  is overwritten and relabelled attack; the rest of the substrate stays normal.
* Target IDs are selected from **calibration** information only: IDs with
  ≥ `min_calib_windows` normal windows and `attack_frac == 0` in calib (top-k by
  calib count). No test data enters target selection.

### 3.3 Comparisons

1. **Normal traffic** — FPR on untouched test normals.
2. **Existing / non-adaptive attack** — recall on real HCRL test attack windows.
3. **Fixed-severity white-box adaptive** — pre-registered grid
   `α ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}` × `direction ∈ {+,-}`.
4. **Gradual / slow-drift adaptive** — α ramps `0 → α_end` over a horizon with
   fast/medium/slow drift rates and linear/exponential shapes.

### 3.4 Pre-registered experiment grid

Locked before running; not adjusted after seeing results. Primary severity grid
as above. Gradual drift: rates `{fast=0.25, medium=0.5, slow=1.0}` × horizon
`300` windows × α_end from the same severity grid × both directions.

### 3.5 Detection boundary & latency

* Detection probability = fraction of injected windows flagged, with Wilson
  95% CIs. `α_50`, `α_90` are the interpolated severities where detection
  probability crosses 50% / 90%; reported as `None` with an explicit reason if
  the grid does not bracket the crossing (floor/ceiling).
* Boundary table per severity: detection probability, precision, recall, FPR,
  F1, PR-AUC (where defined), median + p95 detection latency.
* Fixed-attack latency is degenerate (all windows attacked at once); **latency
  vs severity is measured from gradual-drift runs** (first alarm index after
  attack start), and clearly labelled as such.
* Gradual drift also records max residual norm, cumulative anomaly score,
  detection probability, and the α trajectory.

### 3.6 Leakage controls

```
CALIB normals ─► fit (μ,σ) ─┐
TRAIN normals ─► fit IF ─────┼─► FREEZE ─► generate attack (uses μ,σ only) ─► TEST normals
TRAIN val ─────► threshold ─┘
```

* `AttackParameters` carries the split/id it was fitted from; a provenance
  guard rejects applying test-fit stats.
* Unit tests assert: attack generation touches only target-ID rows; severity 0
  is identity; bounds respected; same seed → same attack; drift monotonic and
  correctly timed.

---

## 4. Files to modify / create

**New library modules (non-destructive):**

* `src/canguard/attacks/__init__.py`
* `src/canguard/attacks/adaptive.py` — attack config, bounds, `AdaptiveLegitIDAttacker`,
  fixed + gradual generation, schedules, provenance guard.
* `src/canguard/evaluation/boundary.py` — detection probability / Wilson CI /
  `α_50`,`α_90` / latency aggregation.

**New experiment runner + configs:**

* `experiments/runners/run_phase2.py` — modes `baseline`, `adaptive`, `gradual`, `all`.
* `experiments/configs/phase2_baseline.yaml`
* `experiments/configs/phase2_adaptive.yaml`
* `experiments/configs/phase2_gradual.yaml`

**New tests:**

* `tests/test_attack_adaptive.py`
* `tests/test_attack_gradual.py`
* `tests/test_boundary.py`

**Docs:**

* `PHASE2_PLAN.md` (this file)
* `ADAPTIVE_ATTACK.md`

**Outputs (created at run time):**

```
experiments/phase2/
  baseline/    config.yaml, metrics.csv/json, predictions.csv, summary.json
  adaptive_attack/ config.yaml, severity_sweep.csv, boundary_table.csv, runs/
  gradual_drift/   config.yaml, drift_results.csv, trajectories.csv, runs/
  results/     consolidated CSVs + summary.json
  plots/       ... (also figures/phase2/)
```

**Not touched:** `features/per_id.py`, `features/window.py`,
`detectors/*`, `evaluation/evaluation.py`, `evaluation/threshold.py`,
`run_pipeline.py`, existing configs/runners.

---

## 5. Baseline command

```bash
python -m experiments.runners.run_phase2 --config experiments/configs/phase2_baseline.yaml --mode baseline
```

Equivalent existing-code check (no scores dumped):

```bash
python -m experiments.runners.train_detector --config experiments/configs/phase2_baseline.yaml
```

## 6. Subsequent commands

```bash
python -m experiments.runners.run_phase2 --config experiments/configs/phase2_adaptive.yaml --mode adaptive
python -m experiments.runners.run_phase2 --config experiments/configs/phase2_gradual.yaml  --mode gradual
```

## 7. Explicitly out of scope this pass

Hybrid PIRD+global detector, DoS/fuzzing/new-ID detectors, cross-dataset work,
streaming, new residualization, window-size sweep. PIRD is the control and is
left unchanged.
