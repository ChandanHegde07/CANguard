# FINAL REPORT — Phase C  
## Scientific Validation & Paper Finalization

**Date:** 2026-08-03  
**Goal:** Reviewer-proof evidence for the residual **representation** contribution (not a new detector).  
**Command:**

```bash
python -m experiments.runners.run_phase_c --config experiments/configs/phase_c.yaml
```

**Runtime:** ~31 minutes (efficient multi-seed: seed aggregation for multi-seed CIs; block bootstrap reserved for paired statistical tests).

---

## 1. Executive summary

Phase C strengthens Phase A/B with multi-seed evaluation, paired statistical tests, feature importance/ablation, runtime/latency, error catalogs, coverage-boundary summary, IEEE-oriented figures, and a representation-first manuscript draft.

| Claim | Phase C support |
|-------|-----------------|
| Residualization improves many detectors on HCRL | **Yes** — multi-seed mean ΔF1 > 0 in **21/24** cells |
| Residualization improves many detectors on ROAD | **Yes** — multi-seed mean ΔF1 > 0 in **30/36** cells |
| Not a single-seed artifact | **Yes** — 5 seeds; seed mean ± std + 95% SE-based CIs |
| Statistically supported deltas | **Mostly yes** — 51/60 F1 paired bootstrap CIs exclude 0 in the positive direction; McNemar often significant when residual wins on decisions |
| Lightweight / practical scoring | **Yes with caveats** — HBOS/EE/AE score very fast; LOF slowest; IF moderate |
| Coverage boundary explicit | **Yes** — strong on targeted legitimate-AID; weak on DoS / fuzzing |
| Reproducible | **Yes** — one CLI, configs, seeds, cached features, tables under `tables/` |

**Contribution framing (unchanged, evidence-backed):**

> Behavioral residualization is a CAN-specific representation that consistently improves multiple unsupervised anomaly detectors under legitimate arbitration-ID reuse, while novel-ID floods and cross-ID fuzzing remain open failure modes.

---

## 2. Experimental additions (what Phase C implemented)

| Task | Deliverable |
|------|-------------|
| Multi-seed (5 seeds: 0–4) | `tables/multiseed_results.csv` (600 rows) |
| Seed-level CIs | `tables/confidence_intervals.csv` (120 rows) |
| McNemar + paired block-bootstrap Δ | `tables/statistical_tests.csv` (120 rows) |
| Permutation importance | `tables/feature_importance.csv` |
| Leave-one-feature-out | `tables/feature_ablation.csv` |
| Runtime / throughput / model size | `tables/runtime.csv` |
| Detection latency | `tables/latency.csv` |
| Top FP/FN catalog | `tables/error_analysis.csv` |
| Coverage summary | `tables/robustness_summary.csv` |
| Figures | `figures/representation/`, `feature_importance/`, `runtime/`, `latency/` |
| Paper draft | `paper/revised_paper.tex` + `paper/updated_tables/` |

**Protocol (identical for all detectors):**  
normal-only train → ~1% FPR threshold on val normals → same splits for raw vs residual → HCRL temporal 40/20/40; ROAD pre-injection train / post-start test.

**SHAP:** Not included. Isolation Forest lacks native TreeSHAP path in our wrapper; KernelSHAP over 14 residual dims × large test sets is costly and would require a surrogate model (Phase A plan). We report permutation + LOFO instead and document the gap.

---

## 3. Multi-seed robustness

### Setup
- Seeds `{0,1,2,3,4}`
- HCRL: DoS, Fuzzy, RPM, gear  
- ROAD: 6 attack-family captures (1 each)  
- Detectors × representations × seeds → **600** completed runs  

### Aggregate residual − raw (using seed **means**)

| Corpus | Cells with mean ΔF1 > 0 | Median mean ΔF1 | Cells with mean ΔROC > 0 | Median mean ΔROC |
|--------|-------------------------|-----------------|---------------------------|------------------|
| HCRL | **21 / 24 (87.5%)** | **+0.47** | **21 / 24** | **+0.10** |
| ROAD | **30 / 36 (83.3%)** | **+0.26** | **28 / 36** | **+0.13** |

These align with Phase A/B single-seed conclusions and show **stability across seeds**.

### Example seed uncertainty
- **RPM residual Isolation Forest:** F1 mean high with **small seed std** (near-ceiling spoofing case).  
- **DoS residual Isolation Forest:** F1 remains near floor across seeds (failure is stable, not noise).

Full per-cell mean/std/CI: `tables/confidence_intervals.csv`.

---

## 4. Statistical findings

### Methods (and why)

1. **Multi-seed mean ± std + normal approx 95% CI across seeds**  
   Addresses stochastic training (IF, AE, subsampled OCSVM/LOF/EE).

2. **Block-bootstrap CI on paired metric deltas (residual − raw)**  
   Same test windows; contiguous blocks reduce IID window assumptions.

3. **McNemar’s test (continuity-corrected)** on paired correctness of residual vs raw predictions  
   Appropriate for **paired binary decisions** on identical instances; tests discordant errors, not independent samples.

We **do not** treat successive CAN windows as IID Bernoulli trials for primary inference.

### Results (seed 0 paired tests)

| Quantity | Value |
|----------|--------|
| Statistical test rows | 120 (metric ∈ {F1, ROC} × tasks × detectors) |
| F1 paired Δ with CI entirely > 0 | **51 / 60** |
| McNemar p < 0.05 with positive F1 Δ | **50** (discordant decisions favor residual in many wins) |
| Negative F1 deltas retained | Counted in table (not dropped) |

Interpretation: for most spoofing / targeted-AID cells, residual gains are supported by both seed aggregation and paired tests. DoS and some AE/EE/OCSVM exceptions remain non-gains or regressions.

---

## 5. Feature importance (why residualization helps)

**Detector:** Isolation Forest on residual features  
**Tasks:** HCRL RPM, gear; ROAD correlated_signal, reverse_light_off  

### Permutation importance (ΔROC-AUC when feature permuted on test)

On **RPM**, top residuals include:

| Rank | Feature | Role |
|------|---------|------|
| 1 | `iat_min_res` | Timing compression / injection cadence |
| 2 | `byte_mean_res` | Payload level shift under spoof |
| 3 | `byte_max_change_res` | Payload dynamics |
| 4–5 | `byte_var_res`, `byte_entropy_res` | Payload distribution |

**Interpretation:** Spoofing detection is driven by **timing + payload residuals**, not a single magic feature. Leave-one-out ablation (`feature_ablation.csv`) ranks features by ΔROC/ΔF1 when removed and retrained—consistent with multi-feature dependence (no single feature explains all gains).

Figures: `figures/feature_importance/perm_importance_*.png`.

---

## 6. Runtime analysis

From `tables/runtime.csv` (residual features, scoring throughput varies by test size; order-of-magnitude comparison):

| Detector | Relative scoring speed | Complexity (qualitative) |
|----------|------------------------|---------------------------|
| Elliptic Envelope / HBOS / AE | Highest windows/s | O(d)–O(d²) / O(d) / small MLP |
| Isolation Forest | Moderate | O(T log ψ) per score |
| One-Class SVM | Lower | O(n_SV) RBF |
| LOF | Lowest | kNN novelty scoring |

**Model size:** IF and kernel models larger than HBOS; exact bytes in `runtime.csv`.

**Embedded discussion (honest):**  
Scoring residual vectors with HBOS/IF is plausible for gateway-class CPUs at automotive message rates **if** feature extraction is optimized (current Python window loop is the bottleneck, not always the sklearn score). LOF is a poor fit for tight latency budgets. We do **not** claim ECU-class deployment without C/Rust reimplementation and hardware measurement.

Figures: `figures/runtime/runtime_train_score.png`, `runtime_throughput.png`.

---

## 7. Latency analysis

Detection latency = first attack window → first true positive (residual features).

| Statistic | Approx. (from `latency.csv`) |
|-----------|-------------------------------|
| Mean delay | ~**23 ms** (timestamp delta) when detected |
| Frames | Often **0** (first attack window already above threshold on strong spoofing cases) |
| Misses | Present for hard tasks (DoS, weak detectors)—not hidden |

Strong spoofing cases often fire immediately at the labeled attack onset (window policy `"any"`).  
Fuzzing/DoS miss rates remain the real reliability issue—not average latency on easy wins.

Figure: `figures/latency/latency_hist.png`.

---

## 8. Error analysis

`tables/error_analysis.csv` (~1000+ rows): top FP / near-FP / FN with:

- CAN ID  
- Score vs threshold  
- Top-3 residual magnitudes  
- Heuristic explanation tags (`benign_iat_shift`, `missed_attack`, payload-dominated, etc.)

**Qualitative patterns:**

| Pattern | Typical setting |
|---------|-----------------|
| FN + mild residuals | DoS novel IDs / global fallback; fuzzing absorbed by per-ID norms |
| FP + IAT residual spikes | Benign driving-state cadence changes (ROAD) |
| FP + payload residuals | Normal payload variation near threshold |
| High residual FPR | Operating point (~1% train FPR) under test shift |

---

## 9. Robustness / coverage boundary

`tables/robustness_summary.csv` + `figures/representation/coverage_boundary.png`

| Family | Residual behavior |
|--------|-------------------|
| Targeted legitimate AID (RPM/gear, reverse light, correlated signal, speedometer) | Strong multi-detector gains |
| HCRL Fuzzy | Intermediate (recall high, FPR often elevated) |
| Novel-ID flood (DoS) | Residual does **not** solve detection at 1% FPR |
| Cross-ID fuzzing (ROAD) | Residual F1 sometimes up; ROC can worsen; not “solved” |

This is the paper’s honest **coverage boundary**.

---

## 10. Scientific conclusions

1. **H1 (residual helps multi-detector)** remains **supported** under multi-seed evaluation on HCRL and ROAD.  
2. Gains are **not** an Isolation Forest story—HBOS, LOF, OCSVM, EE, AE participate.  
3. Failures are **stable across seeds** (DoS), not bad luck.  
4. Mechanism is consistent with **timing + payload residual** importance on spoofing.  
5. Deployment claims should emphasize **score-path lightness** of simple residual models and admit feature-extraction engineering remaining.

---

## 11. Remaining limitations (explicit)

- No cross-vehicle / CAN-FD validation  
- Offline calibration (main path); no streaming residual adaptation in Phase C tables  
- ROAD: one capture per type; frame caps  
- HCRL presence artifacts still inflate absolute scores  
- Single hardware class for runtime  
- AE under-trained (max_iter warnings)  
- SHAP not included  
- Window label policy `"any"`  
- Feature extraction still Python-slow  

These are future work, not hidden defects.

---

## 12. Success criteria checklist

| Criterion | Met? |
|-----------|------|
| 1. Residualization improves multiple detectors on HCRL + ROAD | **Yes** (multi-seed) |
| 2. Improvements statistically supported | **Yes** (seeds + paired bootstrap/McNemar; with exceptions retained) |
| 3. Lightweight enough for practical deployment discussion | **Partially yes** (scoring yes for HBOS/IF; full online stack not proven) |
| 4. Coverage boundary defined | **Yes** |
| 5. Reproducible pipeline | **Yes** |

**arXiv readiness:** Suitable for a representation-focused preprint after light polish of `paper/revised_paper.tex` (fill numeric tables from CSVs, tighten related work).  
**IEEE workshop:** Close—add 1–2 pages of related work + polished figures.  
**IEEE conference:** Needs fuller related work, more ROAD replicates, optional online residual section.

---

## 13. Reviewer #2 simulation (Phase C)

### R1. Novelty  
“Per-ID z-scores are trivial.”  

**Response:** We do not claim inventing residualization or IF. Contribution is **multi-detector, multi-dataset evidence** that a CAN residual representation is the active ingredient under legitimate-ID reuse, plus explicit failure modes. No “first” claim.

### R2. Statistical validity  
“Windows are dependent; seeds are not enough.”  

**Response:** Primary multi-seed CIs are across **independent trainings**. Paired comparisons use **block bootstrap** and **McNemar on paired decisions**. We still admit residual temporal dependence within blocks is imperfect (not capture-cluster bootstrap for all cells).

### R3. Evaluation methodology  
“HCRL is presence-based; ROAD sample is thin.”  

**Response:** ROAD is primary for legitimate-ID claims; HCRL documents DoS failure and presence limits. ROAD uses **pre-injection** protocol; only one capture per type—limitation stated.

### R4. Baseline fairness  
“Did you retune residualization per detector?”  

**Response:** No. Fixed feature set, fixed residual recipe, fixed FPR target, shared splits. Detector hyperparams fixed a priori (subsampling for tractability disclosed).

### R5. Deployment claims  
“Windows/s on a laptop ≠ automotive ECU.”  

**Response:** We report relative detector cost and argue residual **scoring** is light; we **do not** claim production ECU certification. Feature extraction engineering remains.

### R6. Negative results  
“AE/OCSVM sometimes worse after residual.”  

**Response:** Kept in tables; discussed (raw already strong; threshold geometry; reconstruction on whitened residuals).

---

## 14. Reproduce everything

```bash
pip install -e ".[dev]"
# Core science
python -m experiments.runners.run_phase_a --config experiments/configs/phase_a.yaml
python -m experiments.runners.run_phase_b_road --config experiments/configs/phase_b_road.yaml
# Validation package
python -m experiments.runners.run_phase_c --config experiments/configs/phase_c.yaml
python -m pytest tests/ -q
```

**Hardware (this run):** Windows workstation, Python 3.12, scikit-learn 1.7.1, NumPy 2.2.6.  
**Seeds:** 0–4.  
**Configs:** `experiments/configs/phase_{a,b_road,c}.yaml`.

---

## 15. Deliverables map

```
tables/
  multiseed_results.csv
  confidence_intervals.csv
  statistical_tests.csv
  feature_importance.csv
  feature_ablation.csv
  runtime.csv
  latency.csv
  error_analysis.csv
  robustness_summary.csv

experiments/
  multiseed/  statistics/  feature_importance/
  runtime/  latency/  error_analysis/
  phase_c/runs/<timestamp>/

figures/
  representation/  feature_importance/  runtime/  latency/

paper/
  revised_paper.tex
  updated_figures/
  updated_tables/

FINAL_REPORT_PHASE_C.md   # this file
```

---

*Phase C complete. Hypothesis remains supported with multi-seed and paired statistical evidence; failure modes retained; manuscript draft is representation-first for arXiv polishing.*
