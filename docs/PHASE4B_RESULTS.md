# Phase 4B — Representation Density and Transferability: Results

Status: Experiment A (scoring comparison) complete; staged-diversity check run.
Results read from `experiments/phase4b/results/`. Phase 4's negative result is
unchanged and is not tuned away. Full test suite: **185 passed**.

## 1. Method

One frozen model per (source, backbone) was trained with the **identical** Phase 4
protocol (`src/canguard/foundation`), then the **same causal representations** were
scored several ways:

| score | what it is |
|---|---|
| `next_event` | Phase 4 causal next-event residual (the old primary score) |
| `mahalanobis` | regularized full-covariance Mahalanobis (shrinkage 0.1) |
| `mahalanobis_diag` | diagonal Mahalanobis |
| `centroid` | L2 distance to the source-normal mean |
| `knn(k)` | mean distance to k ∈ {1,5,10,20} source normals |
| `probe` | supervised logistic-regression diagnostic (not a detector) |

Density statistics and the representation standardizer are fitted on **source
validation normals only**; inference representations are **causal** (online).
Oracle TPR@1/5%FPR uses a target-normal threshold and is a diagnostic; the
source-threshold operating point is the zero-shot result. 288 result rows
(3 sources × 2 backbones × {in-domain val captures, 5 zero-shot targets} × scores).

## 2. Scoring comparison — zero-shot (mean over transfers)

| score | TPR@1% | TPR@5% | actual FPR | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| next_event | 0.040 | 0.106 | 0.544 | 0.459 | 0.484 | 0.492 |
| mahalanobis | 0.026 | 0.109 | 0.597 | 0.467 | 0.526 | 0.522 |
| mahalanobis_diag | 0.023 | 0.094 | 0.123 | 0.174 | 0.504 | 0.495 |
| centroid | 0.023 | 0.094 | 0.123 | 0.174 | 0.504 | 0.495 |
| knn(1) | **0.052** | 0.149 | 0.685 | 0.571 | 0.525 | 0.540 |
| knn(5) | 0.043 | 0.146 | 0.580 | 0.511 | 0.527 | 0.539 |
| knn(10) | 0.039 | 0.135 | 0.437 | 0.409 | 0.524 | 0.533 |
| knn(20) | 0.036 | 0.129 | 0.307 | 0.304 | 0.520 | 0.527 |

## 3. Scoring comparison — in-domain (held-out source captures)

| score | TPR@1% | TPR@5% | actual FPR | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| next_event | 0.248 | 0.293 | 0.037 | 0.398 | 0.649 | 0.682 |
| mahalanobis | **0.302** | 0.349 | 0.319 | 0.470 | 0.774 | 0.738 |
| mahalanobis_diag | 0.029 | 0.261 | 0.034 | 0.155 | 0.703 | 0.729 |
| centroid | 0.029 | 0.261 | 0.034 | 0.155 | 0.703 | 0.729 |
| knn(1) | 0.256 | 0.345 | 0.372 | 0.504 | 0.776 | 0.739 |
| knn(5) | 0.139 | 0.349 | 0.113 | 0.438 | 0.717 | 0.714 |
| knn(10) | 0.019 | 0.330 | 0.063 | 0.259 | 0.696 | 0.706 |
| knn(20) | 0.020 | 0.226 | 0.059 | 0.179 | 0.679 | 0.701 |

**Finding (Experiment A).** On the *same frozen representations*, density scoring
does **not** beat the next-event residual in zero-shot (best is knn(1) TPR@1%
0.052 vs 0.040, but at actual FPR 0.69 vs 0.54). In-domain, Mahalanobis/kNN
modestly improve the oracle TPR@1% (0.30/0.26 vs 0.25) at higher actual FPR — so
the representation carries *some* attack information that next-event underuses,
but it does not transfer. **The scoring rule is not the primary bottleneck.**

## 4. Representation shift and separability

Means over zero-shot transfers (all backbones):

| quantity | zero-shot | in-domain |
|---|---|---|
| source↔target normal mean |SMD| | 0.485 | 0.457 |
| RBF MMD | 0.392 | 0.386 |
| attack/normal distance ratio | 1.388 | 1.372 |

Per-transfer mean |SMD| (source × target):

|  | Spark | GEM | HCRL | Kia | ROAD | Sonata |
|---|---|---|---|---|---|---|
| HCRL | 0.39 | 0.61 | — | 0.41 | 0.43 | 0.40 |
| Sonata | 0.41 | 0.59 | 0.40 | 0.40 | 0.42 | — |
| GEM | 0.55 | — | 0.56 | 0.57 | 0.57 | 0.56 |

Two facts stand out: (i) shifted representations are **large** (SMD ≈ 0.5)
across all transfers; (ii) the shift is **almost as large in-domain** (0.457) as
zero-shot — the representation is strongly capture/regime-specific, not just
cross-dataset-specific. Separability is weak: the attack/normal distance ratio is
≈ 1.4 (attacks only ~40 % farther from the normal centroid than normal points
are from each other), and ≈ 1.0 for several transfers.

## 5. Mechanism classification

| mechanism | evidence | verdict |
|---|---|---|
| **Case A — representation shift** | SMD ≈ 0.5 zero-shot *and* 0.46 in-domain; MMD ≈ 0.39 | **Dominant** |
| **Case B — weak attack separation** | attack/normal ratio ≈ 1.4; ≈1.0 for many cells | **Contributing** |
| **Case C — scoring failure** | density ≈ next-event zero-shot; no method rescues | **Ruled out as primary** |
| Calibration/threshold shift | actual FPR up to 1.0 at the source threshold | Present (aggravates A) |

The failure is therefore **not** primarily the anomaly score. It is that the
learned representation is not domain/capture invariant, and its normal/attack
geometry is weak even in-domain.

## 6. Decision

**RED for the strong zero-shot foundation-model hypothesis on the available
datasets at this scale**, with an explicit mechanism: this is **Outcome C**
(all scores fail and source→target representation shift is large), not Outcome A.

The density experiment did what it was designed to do: it **exonerated the
scoring rule**. Density scoring extracts slightly more in-domain information than
next-event (Mahalanobis TPR@1% 0.30 vs 0.25), so the representation is not
information-free — but that information is entangled with capture/vehicle-specific
statistics and does not transfer. Proceeding to a serious foundation-model study
on this representation would require a **domain-invariance objective**, not a
better score.

## 7. Staged-diversity check (secondary question)

Three source sets of increasing diversity, same architecture/objective/window
budget (≤20k windows), target Chevrolet Spark (never in any source) and GEM-CAN.
`ndom` = number of source domains.

| source set | ndom | target | TPR@1% | TPR@5% | actual FPR | F1 | mean\|SMD\| | attack ratio |
|---|---|---|---|---|---|---|---|---|
| [sonata] | 1 | Spark | 0.038 | 0.115 | 0.134 | 0.272 | 0.616 | 1.016 |
| [sonata, kia_soul] | 2 | Spark | 0.030 | 0.080 | 0.022 | 0.085 | 0.671 | 1.017 |
| [hcrl, road, sonata, kia_soul, gem_can] | 5 | Spark | 0.011 | 0.073 | 0.047 | 0.123 | 0.545 | 1.031 |
| [sonata] | 1 | GEM | 0.000 | 0.000 | 1.000 | 0.247 | 0.837 | 1.463 |
| [sonata, kia_soul] | 2 | GEM | 0.000 | 0.002 | 1.000 | 0.247 | 0.841 | 1.436 |

Increasing diversity did **not** improve zero-shot TPR@1% (it fell from 0.038 to
0.011 on Spark) and left the representation shift roughly unchanged (0.62–0.67 →
0.55). GEM-CAN remained at FPR = 1.0 in every stage. This is consistent with the
main finding: the bottleneck is the lack of a domain-invariant representation,
which simply concatenating more heterogeneous data does not fix at this scale.
(The comparison is partly confounded by the fixed window budget and unequal domain
sizes; a fully balanced, capacity-controlled sweep remains future work.)

## 8. Limitations and next steps

* Model scale is small (d128, 4 layers, 5 epochs, ≤30k windows); "RED" is scoped
  to this scale and to these four datasets.
* Only the next-event objective was pretrained in Experiment A; masked/contrastive
  objectives and context-length/model-size sweeps were not run.
* Inference representations were causal; a bidirectional representation might
  shift the geometry but would not be streamable.
* The in-domain shift being as large as the zero-shot shift is the key signal:
  the representation is regime-specific. The principled next step is a
  **domain-invariance objective** (e.g., explicit alignment across captures during
  pretraining, or augmentation-based invariance) with a **held-out-capture**
  validation, plus the staged diversity/scale sweep. Adaptive/contextual attacks
  remain deferred until a representation shows nonzero cross-domain separation.
