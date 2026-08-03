# CANguard Final Report

**Status:** Manuscript draft complete with phase A/B/C experimental artifacts.
**Date:** 2026-08-03
**Philosophy:** Optimize for truth, not benchmark numbers. Negative and partial results are first-class outcomes.

This report accompanies the IEEE-style paper at [`papers/canguard_ieee.tex`](papers/canguard_ieee.tex). All numbers below are read directly from `tables/`; none are invented.

---

## 1. Paper Abstract (verbatim from the manuscript)

> Modern vehicles rely on the Controller Area Network (CAN) bus, whose design prioritizes low cost and real-time performance (ISO 11898) and consequently provides no message authentication or encryption. An attacker with physical or remote access to the vehicle network can therefore inject arbitrary frames. Regulatory requirements such as UNECE R155 and ISO/SAE 21434 have made intrusion detection an increasingly important defense-in-depth layer.
>
> Most published CAN intrusion detection systems are built on presence-based features... yet they fail against the realistic threat in which an attacker reuses legitimate IDs. We present per-ID behavioral residualization, a CAN-specific representation designed for that more challenging setting. Our central claim is that this representation, not any single detector, is what improves detection.
>
> Across six unsupervised detectors and two corpora, residualization raises F1 in 21 of 24 HCRL cells and 30 of 36 ROAD cells across five seeds, with 51 of 60 paired bootstrap confidence intervals favoring the residual representation. On the synthetic HCRL dataset, the representation reaches F1 ≈ 0.99 for RPM and gear spoofing. On the real ROAD dataset it maintains recall ≥ 0.99 with mean ROC-AUC ≈ 0.99 across correlated-signal, speedometer, and reverse-light attacks that reuse legitimate IDs. Two failure modes are documented and quantified: novel-ID flooding (HCRL DoS, F1 = 0.02) and cross-ID fuzzing (ROAD, F1 = 0.27), both of which fall outside the coverage of per-ID conditioning. These limits are open research problems rather than hidden defects.

## 2. Key Claims (evidence-backed)

| Claim | Evidence (source file) |
|-------|------------------------|
| Residualization improves detection across multiple detectors | 21/24 HCRL + 30/36 ROAD cells, 5 seeds (`tables/hypothesis_verdict.json`, `tables/road_hypothesis_verdict.json`) |
| Improvement is not a single-seed artifact | 5-seed mean±std (`tables/confidence_intervals.csv`) |
| Gains are statistically supported | 51/60 paired bootstrap CIs exclude 0 (`tables/statistical_tests.csv`); verdict **A_supported** |
| Targeted legitimate-AID attacks detected | HCRL F1 0.938±0.040; ROAD ROC-AUC ≈ 0.99 (`tables/robustness_summary.csv`) |
| Novel-ID flooding (DoS) fails | HCRL F1 0.017, ROC 0.784 (`tables/hcrl.csv` paths, `tables/robustness_summary.csv`) |
| Cross-ID fuzzing fails (ROAD) | F1 0.273 (`tables/road` via `tables/representation_results.csv`) |
| Runtime is practical | HBOS/EE/AE fast, IF moderate, LOF slowest; residual adds negligible overhead (`tables/runtime.csv`, `tables/latency.csv`) |

## 3. Contribution framing (enforced in the paper)

> Behavioral residualization is a CAN-specific **representation** that consistently improves multiple unsupervised anomaly detectors under legitimate arbitration-ID reuse, while novel-ID floods and cross-ID fuzzing remain open failure modes.

The paper does **not** claim a new detector, does **not** claim universal/stealth-attack superiority, and does **not** claim raw "Residual + Isolation Forest" as novel.

## 4. Reproducibility Appendix

The following commands reproduce every table and figure in [`papers/canguard_ieee.tex`](papers/canguard_ieee.tex).

```bash
# HCRL residual-Isolation-Forest baseline
python -m experiments.runners.train_detector --config experiments/configs/hcrl.yaml

# ROAD per-capture validation (pre-injection protocol)
python -m experiments.runners.eval_road --config experiments/configs/road.yaml

# Research phases: baselines, representation, ROAD, statistics/runtime/latency
python -m experiments.runners.run_phase_a      --config experiments/configs/phase_a.yaml
python -m experiments.runners.run_phase_b_road --config experiments/configs/phase_b_road.yaml
python -m experiments.runners.run_phase_c      --config experiments/configs/phase_c.yaml

# Feature-group ablation
python -m experiments.runners.run_ablation --config experiments/configs/ablation.yaml

# Full test suite (61 tests)
python -m pytest
```

Hardware/software used: Python 3.12, NumPy 1.26, pandas 2.1, scikit-learn 1.4. Seeds 0–4. Window size 30. HCRL split 40/20/40; ROAD pre-injection per-capture. Data: HCRL via OCSLab (https://ocslab.hksecurity.net/Datasets/car-hacking-dataset); ROAD via Zenodo DOI 10.5281/zenodo.10462795.

## 5. Artifact inventory

- **Paper:** `papers/canguard_ieee.tex` (IEEE two-column, compiles).
- **Tables (paper sources):** `tables/representation_results.csv`, `tables/multiseed_results.csv`, `tables/confidence_intervals.csv`, `tables/statistical_tests.csv`, `tables/feature_importance.csv`, `tables/feature_ablation.csv`, `tables/runtime.csv`, `tables/latency.csv`, `tables/robustness_summary.csv`, `tables/hypothesis_verdict.json`, `tables/road_hypothesis_verdict.json`.
- **Figures:** `figures/representation/coverage_boundary.png`, `figures/latency/latency_hist.png`, `figures/runtime/runtime_train_score.png`, plus the base HCRL/ROAD and ROC figures.

## 6. Known gaps / unsupported items

- **Architecture figure:** the deliverable contract requests a Mermaid-exported pipeline figure. No generated `*.png` exists for it, and `mermaid-cli` (mmdc) is not available in this environment, so that figure is **not included** rather than fabricated. The pipeline is instead documented textually in the Methodology section.
- **`paper/revised_paper.tex`:** referenced in `FINAL_REPORT_PHASE_C.md` but not present in the repo. The authoritative manuscript is `papers/canguard_ieee.tex`.
- **SHAP:** not computed (Isolation Forest lacks native TreeSHAP in our wrapper); permutation + LOFO reported instead (documented in Phase C report).

---

*End of FINAL_REPORT (manuscript-complete snapshot).*
