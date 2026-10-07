# Phase 1 — PIRD Validation Results

First real values from the Phase 1 validation framework
(`experiments/runners/run_phase1_pird_validation.py`). Phase 1 is a **validation
phase, not a new detector**: it checks (a) that per‑ID residualization is what
drives detection, and (b) that it is deployable, on HCRL (DoS, Fuzzy, RPM, gear)
with the 40/20/40 temporal split.

## Run configuration

* Config: `experiments/configs/phase1_pird_validation_short.yaml`
  (`--mode all`).
* `data_dir: data`, **`sample_size: 60000` frames/file** (a bounded head slice,
  matching Phase A/3/4). The committed config's `sample_size: null` (full files)
  is ~27 h at the measured ~850 rows/s feature loop, so a short run was used; a
  larger ~500k config is retained as `phase1_pird_validation_500k.yaml`.
* Isolation Forest (200 trees), FPR target 1%, val holdout 0.2, seed 0.
* Outputs: `experiments/phase1/{cross_condition,global_threshold,residualization,window_sensitivity}/`.

Machine-readable rows: 12 cross-condition cells, 16 global-threshold rows,
20 residualization rows, 20 window rows.

## 1. Residualization strategy comparison (the core test)

Mean over datasets (F1 / ROC-AUC / recall@1%FPR / actual FPR):

| Strategy | F1 | ROC-AUC | recall@1%FPR | actual FPR |
|---|---|---|---|---|
| `raw` | 0.041 | 0.814 | 0.027 | 0.006 |
| `global_z` | 0.041 | 0.814 | 0.027 | 0.006 |
| `per_id_mad` | 0.150 | 0.923 | 0.152 | 0.018 |
| `per_id_z` (PIRD) | 0.607 | **0.966** | **0.746** | 0.059 |
| `per_id_quantile` | **0.728** | 0.949 | 0.697 | 0.015 |

Per-dataset F1:

| Strategy | DoS | Fuzzy | RPM | gear |
|---|---|---|---|---|
| `raw` | 0.000 | 0.163 | 0.000 | 0.000 |
| `global_z` | 0.000 | 0.163 | 0.000 | 0.000 |
| `per_id_z` | 0.017 | 0.473 | **0.992** | 0.946 |
| `per_id_mad` | 0.000 | 0.586 | 0.000 | 0.013 |
| `per_id_quantile` | **0.800** | 0.180 | 0.959 | **0.975** |

**Findings.**
* Residualization is indeed the driver: per-ID references lift mean F1 from 0.041
  (raw) to 0.61–0.73 and ROC from 0.81 to 0.95–0.97, whereas `global_z` is
  **identical to raw** — so it is the *per-ID* reference, not standardization.
* `per_id_quantile` (distribution-free per-ID CDF→normal) is the best on F1
  (0.728) and the only strategy that detects **DoS** (F1 0.800 vs `per_id_z`
  0.017) — a notable finding, at a much lower actual FPR (0.015).
* `per_id_z` keeps the best ranking/deployment recall@1%FPR (0.746).
* `per_id_mad` is the weakest per-ID variant.

## 2. Cross-condition generalization (`per_id_z`, target never used in fitting)

F1 (source → target):

| src \ tgt | DoS | Fuzzy | RPM | gear |
|---|---|---|---|---|
| DoS | — | 0.485 | 0.751 | 0.690 |
| Fuzzy | 0.011 | — | 0.705 | 0.678 |
| RPM | 0.000 | 0.392 | — | **0.907** |
| gear | 0.009 | 0.283 | 0.842 | — |

Deployment recall@1%FPR is high for spoof targets (0.84–1.00) and ~0 for DoS
targets. Actual target FPR is often elevated (0.04–0.36), i.e. thresholds shift
across conditions.

**Findings.** Learned per-ID norms **transfer across conditions for
spoofing/fuzzy** but **not for DoS** (novel-ID flood), and the frozen source
threshold does not hold its nominal FPR on a new condition.

## 3. Global vs per-dataset threshold (deployment realism)

| strategy | threshold | F1 | actual FPR | recall | recall@1%FPR |
|---|---|---|---|---|---|
| `per_id_z` | per-dataset | 0.607 | 0.059 | 0.746 | 0.746 |
| `per_id_z` | **global (pooled)** | **0.639** | **0.011** | 0.613 | 0.746 |
| `global_z` | per-dataset | 0.041 | 0.006 | 0.027 | 0.027 |
| `global_z` | global | 0.038 | 0.009 | 0.021 | 0.027 |

Per-dataset `per_id_z` detail (global vs per-dataset): a single pooled threshold
actually **reduces** actual FPR from 0.059 to 0.011 (because per-dataset
thresholds over-adjust on some conditions) and raises overall F1, but it *lowers
Fuzzy recall* (0.98 → 0.45) while RPM/gear stay strong.

**Finding.** One pooled operating point is more deployable in the FPR sense and
does not need per-session tuning; the cost is recall on the harder Fuzzy class.

## 4. Window-size sensitivity + latency

| ws | mean F1 | mean recall | mean FPR | mean ROC | ms/window | mean delay (ms) |
|---|---|---|---|---|---|---|
| 10 | 0.552 | 0.751 | 0.065 | 0.965 | 0.019 | 0.004 |
| 20 | 0.572 | 0.749 | 0.067 | 0.960 | 0.018 | 0.000 |
| **30** | **0.607** | 0.746 | 0.059 | 0.966 | 0.016 | 0.001 |
| 50 | 0.578 | 0.551 | **0.021** | 0.957 | 0.018 | 0.118 |
| 100 | 0.005 | 0.002 | 0.005 | 0.958 | 0.017 | 0.229 |

**Finding.** `ws=30` is the best default; `ws=50` lowers FPR at a recall cost;
`ws=100` collapses detection (F1 0.005) on this slice. Scoring cost is flat
(~0.017 ms/window) across sizes, so the trade-off is detection quality vs delay
(larger windows delay onset). `ws=30` is justified rather than assumed.

## 5. Caveats

* Bounded head slice (`sample_size=60000`), not full files; absolute numbers would
  shift slightly on the full stream.
* Single seed; block-bootstrap CIs from Phase A are not attached here.
* False-alarms/hour values are large because HCRL window cadence is high
  (~10⁵–10⁶ windows/hour); treat them as relative comparisons, not vehicle-scale
  alarm budgets.
* HCRL is presence-heavy, so absolute spoof F1 is partly a dataset artifact —
  this is why the ROAD/transfer phases exist.

## 6. Plumbing note

`HCRLLoader.load(sample_size=N)` gained a memory-bounded head-read path so large
HCRL files are not materialised in full (full-file parse needs ~3–5 GB, and the
host had ~2 GB free). It is byte-equivalent to `_parse_file(path).head(N)`,
verified by `tests/test_hcrl_loader.py` (10 passed); full suite **212 passed**.
No PIRD math or protocol changed.

## Bottom line

Phase 1 validates the premise: **per-ID residualization — not generic
standardization — produces the gains**, the effect survives across conditions for
spoofing/fuzzy, a single pooled threshold is deployable (FPR-controlled), and
`ws=30` is a sound default. Two honest limits persist: **DoS (novel-ID flood) does
not transfer**, and frozen source thresholds do not hold nominal FPR under
condition shift. `per_id_quantile` is a promising distribution-free variant that
detects DoS where z-scoring does not.
