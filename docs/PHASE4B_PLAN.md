# Phase 4B — Representation Density and Transferability: Plan

Pre-registered before experiments. Phase 1–4 code/results are untouched.

## Motivation

Phase 4 showed a small ID-free causal model with **next-event residual** scoring
gives mean zero-shot F1 0.14 and TPR@1%FPR 0.03, with FPR=1.0 on GEM-CAN. Two
compounding causes were identified: normal-regime shift and weak attack
separation. It is unknown whether the *representation* lacks transferable
information or the *scoring rule* discards it.

## Central question

> Does the learned temporal representation contain transferable information about
> normal CAN behavior that the next-event residual scoring rule loses?

Secondary: does increasing pretraining diversity/scale improve transfer?

## Experiment A — scoring comparison on identical frozen representations

Train one frozen model per (source, backbone) with the Phase 4 protocol, then
compare **on the same representations**:

* `next_event` — Phase 4 causal next-event residual.
* `mahalanobis` — regularized full covariance (shrinkage 0.1), source-only.
* `mahalanobis_diag` — diagonal covariance, source-only.
* `centroid` — L2 distance to source-normal mean (standardized).
* `knn` — mean distance to k ∈ {1,5,10,20} source normals.
* `probe` — supervised logistic-regression **diagnostic** (source-train →
  target-test), never presented as a zero-shot detector.

All density statistics are fitted on **source validation normal** representations
only; representations are standardized by source-only mean/std. Inference uses
**causal** representations (online protocol). kNN reference sets are capped and
seeded.

## Diagnostics

* Representation shift source-normal vs target-normal: centroid distance, mean /
  median / max |SMD|, RBF MMD.
* Separability: within-target-normal distance, target-normal→source-centroid
  distance, target-attack→target-normal-centroid distance, and their ratio.
* Cross-ID: seen vs unseen ID subsets with representation scoring.
* Per-modality: payload-only / payload+timing / timing-only / ID-free (GEM-CAN).

## Mechanism classification

Each transfer is labelled:

* **Case A** — large source↔target normal representation shift;
* **Case B** — normal and attack representations overlap;
* **Case C** — representations separate but the unsupervised score fails.

## Staged follow-ups (only after Experiment A)

Diversity (single dataset → multiple captures → vehicles → datasets), then modest
model scaling, context length, and objective comparison. Adaptive and stealthy
contextual attacks remain deferred.

## Decision criterion

* **GREEN**: density scoring substantially outperforms next-event residual and
  transfer is usable.
* **YELLOW**: some representation transfer but zero-shot detection stays weak —
  name the exact bottleneck.
* **RED**: all scores, diversity and reasonable scaling fail — the strong
  zero-shot foundation-model hypothesis is unsupported on these datasets.

## Metrics

ROC-AUC, PR-AUC, precision, recall, F1, actual FPR, TPR@1%FPR, TPR@5%FPR,
detection delay where meaningful, plus the shift/separability diagnostics above.
If an FPR is catastrophic (e.g. 1.0) it is reported explicitly. Multiple seeds
where a configuration is worth replicating; single-run prototypes are labelled.
