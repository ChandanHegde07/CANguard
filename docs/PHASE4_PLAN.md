# Phase 4 — ID-Agnostic CAN Temporal Foundation Model (CAN-TFM): Plan

Status: research prototype. This document is written **before** the experiments
and is not adjusted to fit results afterwards.

**Milestone 1 complete** (prototype, zero-shot matrix, cross-ID, ID ablation);
measured results and their interpretation are in `docs/PHASE4_RESULTS.md`.
The hypothesis is **not yet supported** at this scale — see that report's
failure analysis before drawing conclusions.

## 1. Motivation (from Phase 2/3)

PIRD models per-ID behavioral residuals and is strong when the deployment
environment resembles calibration and the relevant IDs exist. Phase 2/3 exposed
three limits:

* **A — ID dependence.** HCRL (11-bit IDs) and GEM-CAN (29-bit extended IDs) have
  **zero** ID overlap, so PIRD has no per-ID reference to apply.
* **B — calibration dependence.** Even with overlapping IDs, source-calibrated
  FPR inflates on the target (e.g. Sonata↔Kia Soul recall 0.94–0.98 but FPR
  0.25–0.47).
* **C — marginal modeling.** PIRD asks "is this unusual for this ID?", not "is
  this plausible given recent context and other signals?".

## 2. Hypothesis

> An ID-agnostic temporal model pretrained on diverse **normal** CAN traffic can
> learn transferable behavioral representations enabling zero-shot anomaly
> detection on unseen networks, vehicles, IDs and datasets without target
> calibration.

This is falsifiable. It is unsupported if: zero-shot transfer collapses on
unseen vehicles; performance depends on ID embeddings; target recalibration is
still required; or cross-dataset transfer is no better than PIRD (Phase 4 §30).

## 3. Data

Fixed dataset families only: HCRL, ROAD, GEM-CAN, HCRL Survival (Sonata / Kia
Soul / Spark). A "domain" is a dataset or a Survival vehicle. Frames are placed
in the canonical schema (`timestamp, can_id, dlc, data_0..7, is_attack,
attack_type`) via the existing loaders. **Pretraining uses normal frames only**,
enforced by an automated provenance check that refuses to train if any
`is_attack == 1` frame is in the pretraining corpus.

Per-domain normal sources and attack sources are recorded in a manifest with:
dataset, vehicle, capture, n_frames, n_ids, CAN bit width, ID format, and the
normal/attack frame split.

## 4. Representation (ID-agnostic)

Each CAN frame becomes a fixed-length feature vector:

```
payload (8 bytes -> [0,1]) | dlc/8 | timing | id-repr
```

* **Payload**: byte mode (8 floats, value/255) or bit mode (64 bits).
* **Timing**: fixed transform `log1p(max(dt_ms,0)) / log1p(1000)` — no target
  statistics.
* **ID representation** (`id_mode`), all fixed transforms so unseen IDs remain
  representable:
  * `none` — ID removed (the crucial ID-free ablation);
  * `numeric` — `id / 2^29` (11-bit IDs are a subset);
  * `bit` — 29-bit one-hot vector (leading zeros for 11-bit IDs);
  * `numeric_embed` — numeric scalar projected by a learned linear layer.

Normalization is never fitted on data, so the target domain contributes nothing.

## 5. Model

Tokens = CAN events. Context length `L ∈ {64,128,256,512}`. Backbone options:
MLP, GRU, small Transformer. CAN-TFM = causal Transformer encoder with
sinusoidal positions and per-modality prediction heads (payload, timing, ID).

## 6. Objectives (self-supervised, normal-only)

1. **Causal next-event prediction**: at position `t`, predict token `t`'s
   features from tokens `< t` (one causal forward pass, streaming-compatible).
2. **Masked event modeling**: randomly mask tokens; reconstruct from full
   context (bidirectional).
3. **Combined**.

Contrastive learning is left as future work (documented) because safe
augmentations for CAN are themselves an open question.

## 7. Anomaly scoring

* Primary: **causal next-event residual** — weighted distance between predicted
  and actual payload/timing (and ID if used). Weights are fixed (per-dimension
  standardization by the *source normal* residual std, computed on source only).
* Diagnostic: **representation distance** to the source-normal embedding
  (Mahalanobis to the source mean/cov, source-only).

ID is never part of the primary score, so the detector is ID-agnostic.

## 8. Zero-shot protocol (frozen)

```
SOURCE normal frames  ->  self-supervised pretraining  ->  freeze
SOURCE validation normals -> score -> threshold at 1% source FPR -> freeze
TARGET frames -> score -> compare to source threshold -> evaluate
```

The target is never used for training, fine-tuning, threshold selection,
normalization, feature/architecture selection, or score calibration.
`tpr_at_{1,5}pct_fpr` are **oracle target-normal diagnostics** (thresholds from
target normals), always reported separately from the source-selected operating
point, which is the actual zero-shot result. Actual target FPR is always reported.

## 9. Baselines

* Baseline 0: PIRD (existing Phase 3 results / rerun infrastructure).
* Baseline 1: MLP next-event predictor.
* Baseline 2: GRU next-event predictor.
* Baseline 3/Candidate: small causal Transformer (CAN-TFM).

## 10. Experiments (order matters)

1. **Prototype (milestone 1):** HCRL in-domain — train on HCRL normals, evaluate
   on a held-out HCRL attack capture. Verify the pipeline end-to-end.
2. **Baselines** on the same split (MLP/GRU/Transformer).
3. **Cross-capture** (within ROAD; within GEM-CAN).
4. **Cross-vehicle** (Sonata↔Kia Soul etc.; Spark as low-overlap stress test).
5. **Cross-ID** (train on IDs disjoint from the test capture).
6. **Cross-dataset** (HCRL↔ROAD, HCRL↔GEM-CAN, ROAD↔GEM-CAN).
7. **Ablations**: ID representation, context length, objective, architecture,
   input modalities, and — most importantly — **pretraining diversity**
   (single capture / vehicle / multiple).
8. **Scaling**: small/medium Transformer sizes; report params, latency, memory.
9. **Stealthy contextual attack** and **adaptive attack**: deferred to a follow-up
   once the transfer matrix is established; the reimplementation is specified but
   not run in this milestone.

## 11. Metrics

ROC-AUC, PR-AUC, precision, recall, F1, actual FPR, TPR@1%FPR, TPR@5%FPR,
detection delay (first sustained N-of-M alarm), false alarms per 10⁵ frames.

## 12. Leakage controls

Automated checks + tests: pretraining corpus contains no attack frames; source
and target captures/vehicles are disjoint; no duplicate windows across splits;
source normalization/threshold constants come from source only; the zero-shot
evaluator raises if target data reaches the fitting API.

## 13. Deliverables

`src/canguard/foundation/*`, `src/canguard/evaluation/zero_shot.py`,
`experiments/phase4/{configs,runners,results,plots}`,
`docs/PHASE4_PLAN.md`, `docs/PHASE4_RESULTS.md`, `tests/test_phase4_*.py`.
Existing Phase 2/3 code and results are untouched.

## 14. Honest scope statement

The full §12 matrix, adaptive attacker and scaling sweep cannot all be completed
in one milestone; the plan above fixes the order. Whatever is not run is marked
`not_run` with the reason rather than fabricated.
