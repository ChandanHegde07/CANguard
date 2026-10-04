# Phase 4 — CAN-TFM Results (ID-Agnostic Temporal Foundation Model)

Status: **milestone 1 prototype + first zero-shot measurements**. Every number
below is read from `experiments/phase4/results/`. Where the result is weak or
negative it is reported as such. The plan (`docs/PHASE4_PLAN.md`) was fixed before
the experiments and is not adjusted to fit results.

## 1. What was built

* `src/canguard/foundation/`
  * `tokenizer.py` — fixed, ID-agnostic frame representation
    (payload byte/bit, DLC, timing; ID modes `none|numeric|numeric_embed|bit`).
  * `model.py` — causal Transformer / GRU / causal-context MLP with per-modality
    heads.
  * `objectives.py` — next-event prediction, masked event modeling, combined.
  * `scoring.py` — primary causal next-event residual (source-normal weights) and
    a diagnostic diagonal-Mahalanobis representation score.
  * `train.py`, `checkpoint.py`, `data.py`, `pipeline.py`, `leakage.py`.
* `src/canguard/evaluation/zero_shot.py` — source-threshold zero-shot metrics,
  oracle TPR@FPR diagnostics, N-of-M detection delay.
* `experiments/phase4/runners/run_phase4.py` — modes `manifest`, `prototype`,
  `zero_shot`, `cross_id`, `ablation`; `phase4_figures.py` for matrices.
* `tests/test_phase4_foundation.py` — 18 tests.

No Phase 2/3 code or results were modified. The full suite is **176 passed**.

## 2. Protocol (frozen)

```
source normal frames -> self-supervised pretraining -> freeze
source validation normals -> residual weights + 1% threshold -> freeze
target frames -> score -> source threshold -> evaluate
```

Targets contribute nothing to fitting or thresholds. `tpr_at_{1,5}pct_fpr` are
**oracle target-normal diagnostics**; the source-selected operating point (with
its *actual* target FPR) is the zero-shot number. An automated check refuses to
pretrain if any selected window contains an attack frame.

## 3. Milestone 1 — in-domain prototype (HCRL)

Source = HCRL normal tail; target = HCRL attack-bearing head (temporally disjoint);
3 epochs, L=128, ID-free.

| Backbone | F1 | actual FPR | TPR@1%FPR |
|---|---|---|---|
| MLP | 0.260 | 0.044 | 0.079 |
| GRU | 0.477 | 0.060 | 0.101 |
| Transformer | 0.252 | 0.041 | 0.062 |

The pipeline is correct and the model learns something in-domain; the temporal
GRU is strongest among the small backbones. These are well below PIRD's in-domain
per-ID numbers and are the starting point, not a claim.

## 4. Zero-shot transfer matrix (ID-free, next-event, 5 epochs)

`experiments/phase4/results/zero_shot/zero_shot_results.csv`. 30 cells.

**F1** (source × target):

|  | Spark | GEM | HCRL | Kia | ROAD | Sonata |
|---|---|---|---|---|---|---|
| HCRL | 0.079 | 0.159 | — | 0.248 | 0.080 | 0.265 |
| ROAD | 0.030 | 0.158 | 0.143 | 0.108 | — | 0.066 |
| GEM | 0.173 | — | 0.310 | 0.154 | 0.105 | 0.211 |
| Sonata | 0.066 | 0.157 | 0.210 | 0.264 | 0.098 | — |
| Kia | 0.032 | 0.158 | 0.145 | — | 0.107 | 0.057 |
| Spark | — | 0.157 | 0.165 | 0.179 | 0.104 | 0.121 |

**Actual target FPR** (source-selected 1% threshold):

|  | Spark | GEM | HCRL | Kia | ROAD | Sonata |
|---|---|---|---|---|---|---|
| HCRL | 0.145 | **1.000** | — | 0.157 | 0.230 | 0.026 |
| ROAD | 0.051 | **1.000** | 0.060 | 0.072 | — | 0.037 |
| GEM | **1.000** | — | **1.000** | **1.000** | **1.000** | **1.000** |
| Sonata | 0.104 | **1.000** | 0.046 | 0.082 | 0.178 | — |
| Kia | 0.005 | **1.000** | 0.016 | — | 0.047 | 0.003 |
| Spark | — | **1.000** | 0.032 | 0.052 | 0.078 | 0.020 |

**TPR@1%FPR (oracle diagnostic)**:

|  | Spark | GEM | HCRL | Kia | ROAD | Sonata |
|---|---|---|---|---|---|---|
| HCRL | 0.001 | 0.000 | — | 0.074 | 0.018 | 0.139 |
| ROAD | 0.001 | 0.000 | 0.047 | 0.016 | — | 0.014 |
| GEM | 0.000 | — | 0.000 | 0.000 | 0.008 | 0.000 |
| Sonata | 0.017 | 0.000 | 0.048 | 0.066 | 0.023 | — |
| Kia | 0.032 | 0.000 | 0.058 | — | 0.036 | 0.066 |
| Spark | — | 0.000 | 0.051 | 0.054 | 0.030 | 0.043 |

**Overall:** mean F1 0.144, mean TPR@1%FPR 0.028, mean actual FPR 0.381.
Best cell by TPR@1%FPR: HCRL→Sonata (0.139, F1 0.265, FPR 0.026).

## 5. Cross-ID zero-shot

`experiments/phase4/results/cross_id/cross_id_results.csv`. Source normals fit the
model; the target is restricted to IDs **unseen** in the source.

| Subset | n | mean F1 | mean TPR@1%FPR | mean actual FPR |
|---|---|---|---|---|
| unseen IDs | 15 | 0.088 | 0.006 | 0.458 |
| seen IDs | 16 | 0.144 | 0.028 | 0.172 |

The model is ID-free (`id_mode=none`), so it scores unseen IDs as readily as seen
ones — the **seen/unseen gap is small**, which is the expected architectural
property PIRD lacks. However, absolute detection is weak in both subsets, so this
demonstrates *ID-agnostic scoring*, not *useful zero-shot detection*.

Note: survival ID-overlap counts are inflated by fuzzy-injection random IDs that
coincidentally recur across vehicles; overlap should be recomputed on normal frames
only (a fix for the next milestone).

## 6. ID-representation ablation

`experiments/phase4/results/ablation_id/ablation_results.csv`. Source =
[HCRL, ROAD, Sonata] normals; target = GEM-CAN (ID-disjoint from every source).

| ID mode | F1 | actual FPR | TPR@1%FPR |
|---|---|---|---|
| none | 0.158 | 1.000 | 0.000 |
| numeric | 0.382 | 1.000 | 0.000 |
| bit | 0.158 | 1.000 | 0.000 |

Adding an ID representation does **not** rescue transfer to GEM-CAN; the ID-free
model is not worse than the ID-based ones on the oracle diagnostic (all 0). This
supports the premise that raw ID embeddings are not the solution, but no setting
achieves useful detection here.

## 7. Interpretation against the claim hierarchy

| Claim | Verdict |
|---|---|
| 1. Learns useful temporal representation of normal CAN | **Partially** (in-domain F1 0.48 with GRU; modest) |
| 2. Transfers across captures | **Not demonstrated** |
| 3. Transfers across vehicles | **Not demonstrated** at useful TPR |
| 4. Transfers across unseen CAN IDs | **ID-agnostic scoring yes**, useful detection **no** |
| 5. Transfers across datasets | **No** — GEM-CAN collapses (FPR 1.0) |
| 6. Robust to marginally-normal attacks | **Not tested** (deferred) |
| 7. Meaningful zero-shot detection | **No** at this scale |

**Failure mechanism.** Two effects compound:
1. *Normal-regime shift.* GEM-CAN (as target or source) drives actual FPR to 1.0
   at the source threshold — the same calibration-shift failure that broke PIRD in
   Phase 3. Source-normal residual weights and threshold do not transfer.
2. *Weak attack separation.* Even the oracle TPR@1%FPR (target-normal threshold)
   is ≤0.14 everywhere, so the gap is not merely a threshold problem — the
   representation does not strongly separate target attacks. Underfitting (small
   model, 5 epochs, 30k windows) is a plausible contributor.

This is a scientifically valid early/negative result: a small ID-free next-event
Transformer does **not** overcome the zero-shot transfer problem at this scale,
and its failure is dominated by the same normal-distribution shift plus limited
temporal discrimination. The hypothesis is **not** supported yet; per the plan's
falsification criteria this is reported rather than tuned away.

## 8. Limitations and next steps

* Scale: d_model 128, 4 layers, 5 epochs, ≤30k windows, ≤120k frames/capture. A
  real foundation model needs orders of magnitude more pretraining data and
  longer schedules.
* Objective: only next-event was used in the matrix; masked modeling and
  combined/contrastive objectives are implemented but not yet swept.
* Scoring: only the next-event residual; the representation-density score is
  implemented but not used as the primary detector — a promising direction given
  the threshold-shift failure.
* Pretraining diversity ablation (single vs multi-domain) is specified but not yet
  run; cross-dataset results hint it matters.
* Detection delay was not meaningful for HCRL (attacks are sporadically labelled,
  not contiguous); apply it to ROAD/GEM/Survival floods next.
* Adaptive/contextual attacks (plan §14–15) are deferred until a representation
  with nonzero oracle separation exists.

**Recommended immediate next steps:** (i) increase pretraining scale/diversity and
epochs; (ii) use the representation-density score and normal-only source
calibration of it; (iii) run the masked + combined objective sweep; (iv) build the
stealthy-contextual attack to test whether temporal context helps where marginals
do not. Do not adopt "foundation model" framing until cross-domain TPR@1%FPR is
substantially above zero.
