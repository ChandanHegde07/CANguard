# FINAL REPORT — Phase A

**Date:** 2026-08-03  
**Scope:** P0, P1, Experiment 1, Experiment 2, Experiment 7, Experiment 18  
**Scientific question:**

> Does per-ID behavioral residualization consistently improve unsupervised CAN intrusion detection across multiple anomaly detectors?

**Contribution framing (enforced):**

> Behavioral residualization is a CAN-specific representation for intrusion detection under legitimate arbitration-ID reuse.  
> Isolation Forest is one detector used to evaluate that representation — not the contribution itself.

**Philosophy:** Optimize for truth. Negative results retained. No result tuning to force the hypothesis.

---

## 1. Repository audit

Full audit: [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md). Phase A status summary:

| Item | Status after Phase A |
|------|----------------------|
| Dataset location | HCRL CSVs under `data/` (not `HCRL Car-Hacking/`) |
| ROAD | Not present — Phase A is **HCRL-only** |
| Multi-detector registry | Implemented |
| Raw vs residual harness | Implemented (`run_phase_a`) |
| Block bootstrap CIs | Implemented |
| Hypothesis verdict | Automated + reported |
| Reproducibility | Config + seed + cache + run directory |

### Strengths used in Phase A

- Canonical loaders, temporal split, FPR-targeted threshold protocol  
- Feature + residual pipeline with tests  
- Detector-agnostic evaluation (`train_anomaly_detector`)

### Remaining weaknesses (out of Phase A scope)

- No ROAD / legitimate-ID-reuse primary benchmark yet  
- Feature extraction still slow (`iterrows` window loop); mitigated by cache  
- Single seed for stochastic models (IF, AE, etc.)  
- HCRL presence-based attacks still confound absolute F1 interpretation  

### Bugs fixed during Phase A

- `pyproject.toml` license format (installable editable package)  
- Declared `PyYAML`, `psutil`, `joblib`  
- Vectorized residual transform (same math, faster)  
- Detector `n_jobs=1` default for reproducible timing  

---

## 2. Experimental methodology

### 2.1 Protocol (identical for every detector × representation)

| Stage | Choice |
|-------|--------|
| Data | HCRL DoS, Fuzzy, RPM, gear |
| Sample | First 60,000 frames per file |
| Windows | Per-ID sliding window, `ws=30` |
| Features | 14 behavioral features (v1) |
| Split | Temporal 40% calib / 20% train / 40% test |
| Residual fit | Per-ID μ/σ on **calib normals** only; global fallback; min 20 windows/ID |
| Train | Normal-only train windows |
| Threshold | (1 − 0.01)-percentile of scores on last 20% of train normals |
| Score | Higher = more anomalous |
| Seed | `0` (global + detector `random_state`) |

### 2.2 Representations

| Name | Features |
|------|----------|
| `raw` | Original window features, NaN→0 |
| `residual` | Per-ID z-score residuals of the same features |

Same window table and split indices; only the transform differs.

### 2.3 Detectors (P1)

| Detector | Implementation | Notes |
|----------|----------------|-------|
| Isolation Forest | sklearn | 200 trees |
| One-Class SVM | sklearn + StandardScaler | Subsample train ≤4k |
| LOF (novelty) | sklearn | Subsample train ≤6k |
| HBOS | Pure NumPy histograms | Lightweight |
| Elliptic Envelope | sklearn + scaler | Rank warnings on collinear features |
| Autoencoder | sklearn MLPRegressor | Reconstruction MSE; max_iter=80 |
| Deep SVDD | **Not run** | Deferred as not necessary for Phase A |

Interface: `fit`, `score` / `score_samples`, `predict`, `save`, `load`.

### 2.4 Statistics (Exp 7)

- **Block bootstrap**, block size 50 windows, 400 replicates, 95% CI  
- Applied to test windows in chronological order  
- Reduces (does not eliminate) IID assumptions among successive windows  

### 2.5 Hypothesis rule (Exp 18, pre-registered in runner)

For each (dataset, detector) cell, \(\Delta = \mathrm{metric}(\mathrm{residual}) - \mathrm{metric}(\mathrm{raw})\).

| Verdict | Rule |
|---------|------|
| **A_supported** | ≥70% cells \(\Delta F1 > 0\), median \(\Delta F1 > 0\), ≥60% cells \(\Delta ROC > 0\) |
| **B_partially_supported** | ≥50% \(\Delta F1 > 0\) **or** (median \(\Delta F1 > 0\) and ≥75% spoof cells improve) |
| **C_rejected** | Otherwise |

### 2.6 Reproducibility

```bash
pip install -e ".[dev]"
python -m experiments.runners.run_phase_a --config experiments/configs/phase_a.yaml
```

Artifacts:

```
tables/baseline_results.csv
tables/representation_results.csv
tables/representation_delta.csv
tables/statistics.csv
tables/hypothesis_verdict.json
experiments/baselines/
experiments/representation/
experiments/statistics/
figures/roc/
figures/pr/
figures/representation/
experiments/phase_a/runs/<timestamp>/
```

---

## 3. Results

### 3.1 Experiment 1 — Residual baselines (F1)

| Detector | DoS | Fuzzy | RPM | gear |
|----------|-----|-------|-----|------|
| isolation_forest | 0.017 | 0.473 | 0.992 | 0.946 |
| one_class_svm | 0.017 | 0.524 | 0.893 | 0.955 |
| lof | 0.000 | 0.504 | 0.905 | 0.989 |
| hbos | 0.178 | 0.425 | 0.977 | 0.918 |
| elliptic_envelope | 0.013 | 0.388 | 0.974 | 0.854 |
| autoencoder | 0.017 | 0.370 | 0.896 | 0.961 |

**Observation:** Under residual features, **spoofing attacks (RPM, gear)** are detected strongly by all detectors. **DoS remains failed** for essentially all detectors at the ~1% FPR operating point. Fuzzy is intermediate (higher recall, elevated actual FPR).

### 3.2 Experiment 2 — Raw vs residual (central)

#### Residual − raw ΔF1

| Detector | DoS | Fuzzy | RPM | gear |
|----------|-----|-------|-----|------|
| isolation_forest | +0.017 | +0.310 | +0.992 | +0.946 |
| one_class_svm | −0.003 | +0.111 | +0.893 | +0.955 |
| lof | 0.000 | +0.364 | +0.904 | +0.624 |
| hbos | +0.159 | +0.216 | +0.977 | +0.918 |
| elliptic_envelope | +0.013 | +0.138 | +0.974 | +0.854 |
| autoencoder | −0.002 | +0.328 | **−0.074** | +0.959 |

#### Aggregate Exp 2 statistics

| Quantity | Value |
|----------|-------|
| Cells with \(\Delta F1 > 0\) | **20 / 24 (83.3%)** |
| Cells with \(\Delta ROC > 0\) | **21 / 24 (87.5%)** |
| Median \(\Delta F1\) | **+0.346** |
| Median \(\Delta ROC\) | **+0.115** |
| Spoof (RPM/gear) cells with \(\Delta F1 > 0\) | **11 / 12 (91.7%)** |

#### Raw F1 (for context)

Most detectors score **~0 F1 on raw RPM/gear** at the 1% FPR operating point, while residual scores jump to ~0.85–0.99. This is the main empirical signal that **representation**, not a single algorithm, enables detection under this protocol.

**Notable exception:** Raw **autoencoder** already achieves F1 ≈ 0.971 on RPM; residual AE F1 is slightly lower (0.896) while ROC-AUC remains ~1.0. Residualization is therefore **not universally necessary** for every detector–attack pair, even when ranking quality is high.

#### Cells where residualization did **not** improve F1

| Dataset | Detector | ΔF1 | ΔROC | Interpretation |
|---------|----------|-----|------|----------------|
| DoS | autoencoder | −0.002 | −0.248 | DoS fails either way; residual can hurt ranking |
| DoS | one_class_svm | −0.003 | −0.195 | Same |
| DoS | lof | 0.000 | −0.113 | No F1 gain; ROC worse |
| RPM | autoencoder | −0.074 | +0.006 | Raw AE already strong; F1 OP sensitive |

These negative cells are **kept** and count against a blanket “always helps” claim.

### 3.3 Runtime / memory / model size (representative residual)

Resource columns are in `tables/baseline_results.csv` (`runtime_seconds`, `peak_rss_mb`, `model_bytes`). HBOS is fastest; LOF scoring is slower; IF/OCSVM/AE moderate. Absolute numbers are machine-specific (single Windows host, seed 0).

---

## 4. Confidence intervals (Exp 7)

Block bootstrap (n=400, block=50, 95% CI) on test windows. Examples:

| Setting | Metric | Point | 95% CI |
|---------|--------|-------|--------|
| RPM, residual, isolation_forest | F1 | 0.992 | [0.989, 0.994] |
| DoS, residual, isolation_forest | F1 | 0.017 | [0.000, 0.040] |

Full CIs for precision, recall, F1, ROC-AUC, PR-AUC, FPR are in `tables/statistics.csv` (`*_ci_low`, `*_ci_high`).

**Interpretation:** High residual F1 on RPM/gear is statistically tight under block resampling. Near-zero DoS F1 is also tightly near zero — failure is not noise.

---

## 5. Scientific findings

### 5.1 Answers to Phase A success criteria

| Question | Answer | Evidence |
|----------|--------|----------|
| 1. Does residualization help? | **Yes, on average and especially for spoofing** | Median ΔF1 = +0.35; 20/24 cells improve F1 |
| 2. Multiple detectors? | **Yes** | IF, OCSVM, LOF, HBOS, EE, AE all gain on Fuzzy/RPM/gear in most cells |
| 3. Statistically significant (narrow sense)? | **Yes for large spoofing gains; DoS failure also significant** | Tight CIs on RPM residual IF F1; DoS CI near 0 |
| 4. Methodology reproducible? | **Yes (HCRL path)** | One-command runner, config, cache, seeds, tables |
| 5. Strong enough to continue Phase B? | **Yes, with caveats** | Hypothesis **A_supported** on HCRL; ROAD still required for legitimate-ID claims |

### 5.2 Exp 18 verdict

**Verdict: A_supported** (`tables/hypothesis_verdict.json`)

Under the pre-registered rule, residualization **consistently improves** unsupervised detection **across detectors** on this HCRL protocol.

### 5.3 What the contribution is (and is not)

| Claim | Supported? |
|-------|------------|
| Residualization is an effective **representation** for multi-detector CAN anomaly detection on HCRL spoofing/fuzzy patterns | **Yes** |
| Residualization + IF is a novel detector | **Not claimed; not needed** |
| Residualization fixes novel-ID DoS flooding | **No** |
| Results prove legitimate-AID reuse (ROAD-style) generalization | **Not yet** (no ROAD in Phase A) |
| Residualization always improves every detector on every attack | **No** (DoS; raw AE on RPM) |

---

## 6. Failed experiments / negative results

1. **DoS (all detectors, residual and raw):** Operating-point detection fails. Residualization does not rescue novel-ID flooding under per-ID baselines + 1% FPR.  
2. **Residual can hurt ranking on DoS** for OCSVM, LOF, AE (negative ΔROC).  
3. **Raw AE on RPM** already strong — residualization is not the only path for that detector.  
4. **Deep SVDD** not evaluated (feasibility deferral, not a hidden negative).  
5. **Elliptic Envelope** rank warnings (covariance not full rank) — results retained but Gaussian assumption stressed.  
6. **Actual FPR** often exceeds 1% target on Fuzzy residual runs — calibration is imperfect under distribution shift.

None of these were removed or re-tuned away.

---

## 7. Limitations

1. **HCRL only** — presence-based attacks; absolute F1 on spoofing is partly a dataset artifact.  
2. **60k-row prefix** — not full-file evaluation.  
3. **Single random seed** for stochastic models.  
4. **Window label policy `"any"`** may inflate attack windows.  
5. **Block bootstrap** still imperfect dependence model (not capture-level; HCRL is one stream).  
6. **No ROAD** — cannot yet claim legitimate arbitration-ID reuse in the strong sense.  
7. **Subsampling** for OCSVM/LOF/AE/EE for tractability.  
8. **AE under-trained** (max_iter=80, convergence warnings).  

---

## 8. Reviewer #2 critique (Phase A)

> **Simulated skeptical review**

**R1. HCRL is too easy / wrong primary benchmark.**  
Near-perfect residual F1 on RPM/gear may rediscover spoof/presence structure. Without ROAD, the “legitimate ID reuse” framing is undersupported.

**R2. “Consistently” overstates DoS.**  
Four detectors show no real DoS benefit; some ROC drops. The hypothesis is only “consistent” under an aggregate rule that tolerates 30% non-improving cells.

**R3. Operating point vs ranking.**  
Large ΔF1 can come from threshold geometry. Raw AE RPM shows residual F1 can drop while ROC stays high — report both always (you do, good).

**R4. Single seed, subsampled detectors.**  
OCSVM/LOF/AE use train caps; IF is stochastic with one seed. Multi-seed variance missing.

**R5. Novelty.**  
Per-ID z-scores are simple. Multi-detector evidence helps, but related residual/anomaly work must be cited carefully; avoid “first.”

**R6. Reproducibility on other machines.**  
Cache and config help; datasets are local large CSVs; exact runtime/memory not portable.

### Responses

| Criticism | Response |
|-----------|----------|
| **R1 HCRL** | **Admission.** Phase A intentionally answers multi-detector residualization on available HCRL data. **ROAD is required before strong legitimate-ID claims.** Proceed to Phase B only with that caveat. |
| **R2 DoS** | **Admission + evidence.** DoS is a documented failure mode; residualization does not solve novel-ID floods. Paper must lead with conditional success (spoof/fuzzy yes, DoS no). |
| **R3 OP vs ranking** | We report F1 **and** ROC/PR + actual FPR. Negative AE RPM ΔF1 retained. |
| **R4 seeds/subsampling** | **Limitation admitted.** Multi-seed is Phase B sensitivity work; Phase A uses fixed seed 0 for comparability. |
| **R5 novelty** | Contribution framed as **representation evidence**, not algorithm invention. No “first” claim. |
| **R6 portability** | One-command runner + tables committed under `tables/`; hardware metrics secondary. |

---

## 9. Phase A success criteria decision

| Criterion | Met? |
|-----------|------|
| Residualization help assessed | **Yes** |
| Multi-detector evidence | **Yes** |
| Statistical intervals | **Yes** (block bootstrap) |
| Reproducible pipeline | **Yes** |
| Evidence to continue Phase B | **Yes, conditional** |

**Recommendation:** Proceed to Phase B (importance, latency, runtime, robustness, transfer, streaming, deployment) **and prioritize ROAD evaluation** so the paper’s legitimate-ID narrative is not HCRL-only.

**Do not** reframe the paper as “Residual Isolation Forest.”  
**Do** state:

> Across six unsupervised detectors on HCRL, per-ID behavioral residualization improved F1 in 20/24 detector–attack cells (median ΔF1 = +0.35), with large gains on RPM/gear spoofing and Fuzzy, while DoS (novel-ID flood) remained unsolved.

---

## 10. How to reproduce

```bash
pip install -e ".[dev]"
python -m experiments.runners.run_phase_a --config experiments/configs/phase_a.yaml
python -m pytest tests/ -q
```

Config: `experiments/configs/phase_a.yaml`  
Data expected at: `data/{DoS,Fuzzy,RPM,gear}_dataset.csv`

---

## 11. Implementation inventory (Phase A)

| Component | Path |
|-----------|------|
| Seeds / cache / logging / tracking | `src/canguard/exp/` |
| Detector registry + 6 detectors | `src/canguard/detectors/` |
| Block bootstrap | `src/canguard/evaluation/bootstrap.py` |
| Resource timing | `src/canguard/exp/resources.py` |
| Matrix builder | `src/canguard/exp/matrix.py` |
| Phase A runner | `experiments/runners/run_phase_a.py` |
| Config | `experiments/configs/phase_a.yaml` |
| Vectorized residuals | `src/canguard/features/per_id.py` |

**Tests:** 58 passed (including new detector registry + bootstrap tests).

---

*End of Phase A final report. Hypothesis evaluated: **A_supported** (with DoS failure and HCRL limitations honestly retained).*
