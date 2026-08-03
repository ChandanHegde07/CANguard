# CANguard Final Report

**Status:** Pre-implementation snapshot (planning complete; experiment suite not executed).  
**Date:** 2026-08-03  
**Philosophy:** Optimize for truth, not benchmark numbers. Negative and partial results are first-class outcomes.

This report will be rewritten after Phases P0–P7. Sections marked **[PENDING]** await real experimental artifacts. Do not cite PENDING numbers as findings.

Related documents:

- [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md) — repository audit  
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) — Exp 1–18 design  

---

## 1. Everything Implemented (current)

### 1.1 Library (exists)

| Module | Status |
|--------|--------|
| HCRL + ROAD loaders | Yes |
| Per-ID window features (14 behavioral) | Yes |
| Per-ID z-score residualization | Yes |
| Temporal splits | Yes |
| Isolation Forest detector | Yes |
| Shared train / FPR-threshold / metrics | Yes |
| Visualization (ROC/PR, distributions, matrix, timeline) | Partial |
| Multi-detector registry | **No** |
| Raw vs residual harness | **No** (notebook only for IF) |
| Runtime / memory / latency modules | **No** |
| Bootstrap / error catalog / streaming residuals | **No** |
| Manuscript (`paper/`) | **No** |

### 1.2 Experiments executed in-repo runners

| Runner | Status |
|--------|--------|
| Residual IF on HCRL | Code exists; **data absent in workspace** |
| ROAD residual IF | Code exists; protocol mismatch vs docs (audit B1–B4) |
| Feature-group ablation | Partial; `exclude_groups` broken |

### 1.3 Planning artifacts (this session)

| Artifact | Status |
|----------|--------|
| `PROJECT_AUDIT.md` | Complete |
| `EXPERIMENT_PLAN.md` (Exp 1–18) | Complete |
| `FINAL_REPORT.md` | This document (pre-results) |
| `tables/`, `paper/`, Exp 1–18 outputs | **Not created** |

---

## 2. Scientific Findings

### 2.1 From prior notebook evidence (HCRL residual IF, not re-run here)

| Finding | Support |
|---------|---------|
| Residual IF strong on RPM/gear spoofing | Notebook metrics F1 ≈ 0.99 / 0.95 |
| Residual IF weak on DoS | Recall ≈ 0.01, F1 ≈ 0.02 |
| Residual IF moderate on Fuzzy | High recall, high FPR, F1 ≈ 0.47 |
| Raw IF ≪ residual IF on RPM/gear | Notebook ablation (IF only) |
| HCRL attacks are presence-separable | EDA trivial rules F1 > 0.99 |
| ROAD targeted AID: high recall (reported) | `docs/road_validation.md` (protocol caveats) |
| ROAD fuzzing: weak | Same doc |

### 2.2 Not yet established

| Claim | Status |
|-------|--------|
| Residualization helps **regardless of detector** | **[PENDING] Exp 2 + 18** |
| Statistically significant residual gains | **[PENDING] Exp 7** |
| Cross-dataset transfer | **[PENDING] Exp 12** |
| Real-time deployability | **[PENDING] Exp 8–9, 13** |
| Per-byte residuals help spoofing | **[PENDING] Exp 14** |
| Quantitative robustness boundaries | **[PENDING] Exp 16** |

---

## 3. Positive Results (so far)

1. **Clear residual representation** formalized and tested for equivalence to research notebooks.  
2. **Honest failure reporting** already in README (DoS, presence-based HCRL limits).  
3. **ROAD integration path** exists (preferred legitimate-ID benchmark).  
4. **Preliminary representation signal:** residual ≫ raw for Isolation Forest on spoofing attacks (notebook).  
5. **Repositioned hypothesis** documented; plan avoids “Residual+IF is novel.”

---

## 4. Negative Results (so far)

1. **DoS:** per-ID residualization + IF fails at ~1% FPR target (novel-ID flood / global fallback).  
2. **Cross-attack transfer within HCRL:** near-zero off-diagonal recall (notebook / pipeline).  
3. **ROAD fuzzing:** residual IF does not generalize well (reported).  
4. **Actual FPR** often exceeds 1% target (Fuzzy, some ROAD types) — calibration brittle.  
5. **No multi-detector confirmation** that residualization is the general contribution.

---

## 5. Failed / Partial Hypotheses

| Hypothesis | Verdict now |
|------------|-------------|
| Residual + IF is a complete IDS for all HCRL attacks | **Failed** (DoS) |
| Residualization helps regardless of detector | **Untested** |
| PIRD generalizes to all ROAD attacks | **Partial** (targeted AID yes-ish; fuzzing no) |
| Near-perfect HCRL scores ⇒ real behavioral IDS | **Failed as implication** (presence shortcuts) |

Exp 18 will re-judge the central multi-detector hypothesis after Exp 1–2–7.

---

## 6. Remaining Weaknesses

See audit for full list. Critical subset:

1. IF-only production path; representation claim unsupported at scale.  
2. ROAD protocol docs ≠ code.  
3. No data/results in workspace; reproducibility for third parties incomplete.  
4. No CIs, multi-seed, resource metrics, error catalog.  
5. Slow `iterrows` residual/feature loops.  
6. No manuscript.  
7. Ablation `exclude_groups` silent no-op.  
8. PyYAML undeclared dependency.

---

## 7. Novel Contributions (target, after improvements)

**If Exp 18 supports residualization broadly:**

1. Per-ID behavioral residualization as a **CAN-specific representation** for legitimate-ID reuse.  
2. Multi-detector evidence that the representation—not a single algorithm—drives gains.  
3. Quantified **failure boundaries** (novel-ID %, cross-ID %, burst length).  
4. Offline vs streaming residualization comparison.  
5. Deployability characterization (latency, RAM, throughput).

**If Exp 18 is partial or negative:**

1. Rigorous **conditional** characterization: when residualization helps vs harms.  
2. Negative multi-detector result is still a contribution if well measured.

Do **not** claim algorithm novelty for Isolation Forest or residual anomaly detection in general ML.

---

## 8. Publication Readiness Scores

Scores are **0–10**. Pre-implementation baseline.

| Venue | Score | Rationale |
|-------|-------|-----------|
| **arXiv preprint** | **3 / 10** | Solid prototype + honest narrative, but no multi-detector study, no paper, no versioned tables, ROAD protocol issues, data not bundled |
| **IEEE workshop** | **2 / 10** | Needs Exp 1–2–7–10 minimum, IEEE figures, threat model, stats, limitations |
| **IEEE conference** | **1 / 10** | Needs workshop bar + transfer/streaming/boundaries + stronger related work + reviewer-proof stats |

### Score trajectory (targets after phases)

| After phase | arXiv | Workshop | Conference |
|-------------|-------|----------|------------|
| P2 (Exp 1–2–7–18 core) | 6 | 4 | 3 |
| P3–P4 (errors, sensitivity, boundaries) | 7 | 6 | 4 |
| P5–P6 (deploy + extensions) | 8 | 7 | 6 |
| P7 (paper + figures polish) | 8–9 | 7–8 | 6–7 |

Scores assume honest reporting; inflated metrics without ROAD protocol fix should **lower** scores.

---

## 9. Reviewer #2 Simulation (current paper/repo as-if submitted)

> **Confidential review — simulated skeptical top-tier reviewer**

### Summary

The authors propose per-ID behavioral residual features for CAN intrusion detection, evaluated primarily with Isolation Forest on HCRL and partially on ROAD. The writing acknowledges some dataset limitations, which is refreshing. However, the core claim appears to conflate a **preprocessing transform** with a **detection method**, experiments are single-algorithm and single-seed, the harder benchmark protocol is inconsistently implemented, and several results on HCRL are confounded by presence-based attacks. I cannot recommend acceptance in the present form.

### Major comments

**M1. Novelty is unclear.** Residualization and Isolation Forest are standard. What is CAN-specific beyond applying z-scores per arbitration ID? Without multi-detector ablation showing the representation is the active ingredient, this reads as an engineering pipeline paper, not a scientific contribution on representations.

**M2. Hypothesis untested.** The abstract-level claim that residualization improves anomaly detection is only shown for IF (and only clearly in a notebook ablation). Please evaluate OCSVM, LOF, etc., under **identical** splits and thresholds. If residualization fails for some detectors, say so.

**M3. HCRL is a weak primary benchmark for behavioral IDS.** The authors themselves note trivial presence rules achieve F1 > 0.99. Then why lead with near-perfect RPM/gear residual-IF scores? This invites the interpretation that the method rediscovers presence/spoof artifacts. ROAD should be primary; HCRL secondary for failure modes.

**M4. ROAD evaluation is not trustworthy as described.** Documentation claims pre-injection calibration; implementation appears to fit stats on all normals and apply a global temporal split that may place injections in train/test arbitrarily. This is a methodological defect. Fix or retract ROAD claims.

**M5. No statistical rigor.** Single `random_state=0`, no confidence intervals, windows treated as IID for any implicit comparison. CAN windows are highly dependent. Use block bootstrap or capture-level aggregation.

**M6. Operating point confusion.** Targeting 1% FPR on train normals often yields much higher test FPR (e.g. Fuzzy). Reporting F1 without stressing actual FPR is misleading for IDS deployment.

**M7. Failures are qualitative.** “DoS fails” is not enough. Quantify novel-ID fraction vs recall, burst length vs performance, cross-ID coverage vs performance. Without a robustness boundary, the threat model is incomplete.

**M8. Deployability unsubstantiated.** “Lightweight” requires latency, RAM, throughput, model size, and preferably streaming residual updates—not only sklearn defaults on a laptop subsample (60k rows).

**M9. Reproducibility.** Datasets gitignored, results gitignored, unpinned dependencies, PyYAML missing from deps, no locked environment, no committed tables. I could not reproduce the paper’s numbers from the repository alone.

**M10. Missing related work depth.** Residual-based anomaly detection, automotive IDS surveys, and ROAD-specific prior detectors need systematic comparison. Avoid “first” language.

### Minor comments

- Label policy `"any"` inflates window-level positives; sensitivity analysis needed.  
- Cross-attack matrix uses target residual stats on source train—explain or it looks like leakage.  
- Empty `transforms/` / incomplete ablations suggest unfinished work.  
- Figures are useful but not IEEE-consistent; no embedding of residual space.

### Recommendation

**Reject / major revision** (as a conference paper). Suitable for arXiv only after multi-detector representation study, ROAD protocol correction, statistical analysis, and full reproducibility package.

---

## 10. Responses to Reviewer #2

| Criticism | Response plan | Status |
|-----------|---------------|--------|
| **M1 Novelty** | Reframe as residual **representation**; Exp 1–2–18 multi-detector; no IF novelty claim | Planned |
| **M2 Hypothesis untested** | Exp 2 factorial + Exp 18 pre-registered verdict | Planned |
| **M3 HCRL primary** | Make ROAD primary tables; HCRL for presence/DoS failure | Planned (paper structure) |
| **M4 ROAD protocol** | Fix pre-injection calib; honor frame limits; re-run; admit prior numbers provisional | **Must fix first (P0)** |
| **M5 Statistics** | Exp 7 block/cluster bootstrap; multi-seed | Planned |
| **M6 Actual FPR** | Always report actual FPR; threshold sweeps Exp 6/15 | Partial (sweep exists) |
| **M7 Qualitative failures** | Exp 10 error catalog + Exp 16 boundary plots | Planned |
| **M8 Deployability** | Exp 8–9–13 | Planned |
| **M9 Reproducibility** | Pin deps; cache; commit tables or release scripts; document data URLs | Planned |
| **M10 Related work** | Expand in `paper/revised_paper.tex` | Planned |
| Label policy | Exp 15 sensitivity | Planned |
| Cross-attack residualization | Document clearly; optional ablation | Planned |

**Admissions that remain until fixed:**

- Current ROAD numbers in README/docs are **provisional** pending protocol fix.  
- Central multi-detector hypothesis is **not yet tested**.  
- Repository alone is **not currently sufficient** for full reproduction without external datasets and re-runs.

---

## 11. Research Philosophy Checklist

| Principle | Commitment |
|-----------|------------|
| Never optimize only for benchmark numbers | Yes |
| Keep experiments that weaken the paper | Yes |
| Report disproven / partial hypotheses | Yes (Exp 18) |
| Credible, reproducible contribution | Target of Phases P0–P7 |

---

## 12. Next Implementation Gate

Proceed only when user requests implementation. Recommended first PR/wave:

1. P0 ROAD protocol + dependency fixes  
2. Detector registry + resource metrics  
3. Exp 2 + Exp 1 matrix → `tables/representation_results.csv`, `tables/baseline_results.csv`  
4. Update this FINAL_REPORT with real Exp 18 verdict  

---

*End of pre-implementation FINAL_REPORT.*
