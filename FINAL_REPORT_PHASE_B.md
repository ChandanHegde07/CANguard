# FINAL REPORT — Phase B (ROAD)

**Date:** 2026-08-03  
**Focus:** Correct ROAD protocol + multi-detector raw vs residual study  
**Question (same as Phase A, harder data):**

> Does per-ID behavioral residualization consistently improve unsupervised CAN intrusion detection across multiple anomaly detectors **under legitimate arbitration-ID reuse (ROAD)**?

**Framing:** Behavioral residualization is the representation under test. Detectors are evaluation tools, not the contribution.

---

## 1. What was done

### 1.1 Protocol fix (audit P0 for ROAD)

Previous ROAD code fitted residuals on **all normals** and used a naive 40/20/40 temporal split (could place injections in train).  

**New protocol (`pre_injection_v1` / frame logic v2):**

| Stage | Rule |
|-------|------|
| Residual stats | Fit **only on pre-injection** windows (`elapsed < injection_start`) |
| Train | Pre-injection windows only (normals) |
| Threshold | Holdout of pre-injection normals (~1% FPR target) |
| Test | Windows with `elapsed >= injection_start` (injection + after) |
| Raw vs residual | Same train/test indices |

Implemented in `src/canguard/exp/road_protocol.py`.

### 1.2 Additional fixes

- ROAD data path: `road/road/`
- Faster ROAD log parse (vectorized hex splits)
- Frame selection prioritizes **injection region** when `max_frames` binds
- Window `elapsed` stays relative to **capture start** (not first loaded frame) — critical for late injections (e.g. speedometer @ 42s)
- Cache key protocol bump to avoid stale windows

### 1.3 Experiment matrix

| Factor | Values |
|--------|--------|
| Captures | 1 per attack type (6 types after speedometer re-run) |
| Detectors | IF, OCSVM, LOF, HBOS, Elliptic Envelope, Autoencoder |
| Representations | raw, residual |
| Metrics | P/R/F1, ROC/PR-AUC, FPR, runtime, model size, block-bootstrap CIs |

```bash
python -m experiments.runners.run_phase_b_road --config experiments/configs/phase_b_road.yaml
# speedometer re-run after elapsed fix:
python experiments/scripts/run_speedometer_fix.py
```

---

## 2. Hypothesis verdict (ROAD)

**Verdict: A_supported** (`tables/road_hypothesis_verdict.json`)

| Statistic | Value |
|-----------|--------|
| Cells (capture × detector) | **36** |
| ΔF1 > 0 | **32 / 36 (88.9%)** |
| ΔROC > 0 | **~80%+** (see tables) |
| Median ΔF1 | **~0.42** (order of magnitude; see CSV for exact) |
| Targeted AID F1 win rate | High (~0.88) |
| Fuzzing F1 win rate | 1.0 on F1 cells (but absolute performance still weak) |

Pre-registered rule (same as Phase A): A if ≥70% ΔF1>0, median ΔF1>0, ≥60% ΔROC>0.

---

## 3. Results summary

### 3.1 Mean residual vs raw F1 by attack type

| Attack type | Mean raw F1 | Mean residual F1 | Pattern |
|-------------|'-------------|------------------|---------|
| correlated_signal | ~0.00 | ~0.45 | Large residual gain; high residual ROC (~0.94) |
| reverse_light_off | ~0.00 | ~0.57 | Strong residual gain; residual ROC ~0.99 |
| reverse_light_on | ~0.15 | ~0.74 | Strong residual gain |
| max_speedometer | mixed raw | residual helps most detectors; **OCSVM F1 drops** | Residual ROC high for IF/HBOS |
| max_engine_coolant | moderate raw | small mean gain; **AE/EE F1 drop** | Residual ROC near 1.0 |
| fuzzing | ~0.13 | ~0.23 | Residual helps F1 slightly; **ROC can worsen** |

Exact numbers: `tables/road_representation_results.csv`, `tables/road_representation_delta.csv`.

### 3.2 Scientific reading

1. **Legitimate single-AID injections (correlated signal, reverse light):** residualization is the main enabler — raw features collapse at the 1% FPR operating point for many detectors; residual yields high recall and strong ROC.

2. **Fuzzing:** residualization does **not** make fuzzing “solved.” F1 stays low–moderate; residual FPR often high; ranking (ROC) can degrade. Matches Phase A intuition: cross-ID bursts stress per-ID baselines.

3. **Not every detector always gains on F1:** documented negatives include:
   - max_engine_coolant: autoencoder, elliptic_envelope (raw already decent F1; residual F1 lower, ROC still high)
   - reverse_light_on: autoencoder (raw F1 already high)
   - max_speedometer: one_class_svm F1 residual &lt; raw

4. **Operating point vs ranking:** residual often improves ROC even when F1 at 1% FPR is middling (calibration / class imbalance / low attack density after injection start).

---

## 4. Confidence intervals

Block bootstrap CIs are stored in `tables/road_representation_results.csv` (`f1_ci_low`, `f1_ci_high`, …). Use them in paper tables for residual IF on reverse_light / correlated_signal (tight when separation is strong).

---

## 5. Phase A + Phase B joint conclusion

| Dataset | Verdict | Residual helps multi-detector? | Hard failure mode |
|---------|---------|--------------------------------|-------------------|
| HCRL (Phase A) | A_supported | Yes (esp. RPM/gear, Fuzzy) | DoS novel-ID flood |
| ROAD (Phase B) | A_supported | Yes (esp. targeted AID) | Fuzzing; some detector-specific F1 regressions |

**Contribution statement supported so far:**

> Per-ID behavioral residualization is an effective CAN-specific **representation** that improves multiple unsupervised detectors under **legitimate-ID** spoofing-style attacks, while remaining weak on novel-ID floods (HCRL DoS) and cross-ID fuzzing (ROAD).

**Still not claimed:** residual+IF novelty; universal improvement for every detector×attack; real-time deployment (not measured beyond train/score times).

---

## 6. Negative results (kept)

1. ROAD fuzzing remains hard (low F1, residual ROC sometimes worse than raw).  
2. Residual can lower F1 when raw is already strong (AE/EE on coolant; AE on reverse_light_on; OCSVM on speedometer).  
3. Actual FPR often exceeds 1% target on residual ROAD (e.g. correlated_signal residual FPR ~0.15–0.35 depending on detector).  
4. max_speedometer initially skipped due to elapsed/frame-selection bugs — fixed and re-run; report this as engineering risk for late injections.  
5. Deep SVDD not evaluated.  
6. Only **one capture per attack type** in this Phase B pass.

---

## 7. Limitations

- `per_type: 1` — not all ROAD replicates  
- `max_frames: 80000` — long captures truncated (injection-priority)  
- Single seed  
- Window label policy `"any"`  
- Feature extraction still slow (row loop); mitigated by cache  
- Masquerade variants not evaluated  

---

## 8. Reviewer #2 (ROAD-focused)

**R1. “You fixed the protocol after seeing numbers.”**  
Protocol was fixed **before** the main multi-detector ROAD run based on audit. Speedometer elapsed bug was found when one capture failed (0 test windows) and corrected; re-run is disclosed.

**R2. “F1 still modest on correlated_signal (~0.45) despite high recall.”**  
True — residual FPR is elevated. Report recall, FPR, ROC, PR together; do not lead with F1 alone.

**R3. “Fuzzing contradicts ‘consistent’.”**  
Aggregate rule can still say A_supported while fuzzing is weak. Paper must say **conditional consistency**: strong on targeted AID reuse; weak on multi-ID fuzz.

**R4. “Only one capture per type.”**  
Valid. Next step: `per_type: null` or 3 for variance across sessions.

### Responses

| Point | Action |
|-------|--------|
| R1 | Disclose bugfix + re-run in paper appendix |
| R2 | Always publish FPR + PR-AUC |
| R3 | Partial/conditional language in abstract |
| R4 | Phase B follow-up: full capture set |

---

## 9. Artifacts

| Path | Content |
|------|---------|
| `tables/road_representation_results.csv` | Full factorial metrics + CIs |
| `tables/road_representation_delta.csv` | Residual − raw deltas |
| `tables/road_baseline_results.csv` | Residual-only baselines |
| `tables/road_hypothesis_verdict.json` | Exp 18-style verdict |
| `figures/representation/road_representation_*.png` | Bar charts (from main run; speedometer may need replot) |
| `experiments/runners/run_phase_b_road.py` | Main runner |
| `src/canguard/exp/road_protocol.py` | Correct protocol |
| `FINAL_REPORT_PHASE_A.md` | HCRL multi-detector study |

---

## 10. What’s next (after Phase B core)

1. **Full ROAD** (`per_type: 3` or all captures) for session variance  
2. **Error analysis / robustness boundary** (fuzzing, novel-ID)  
3. **Latency + runtime** (Exp 8–9)  
4. **Feature importance** (Exp 3–4)  
5. Paper draft with representation-first narrative, HCRL+ROAD tables  

---

*End of Phase B ROAD report. Hypothesis on ROAD: **A_supported**, with fuzzing and detector-specific F1 regressions retained as negative evidence.*
