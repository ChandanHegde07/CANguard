# Phase 2 — External (independent) validation: status and interface

**Status: COMPLETE (frozen).** ROAD external validation was run with the frozen
methodology; results are in `experiments/phase2/road_external/` and summarised in
`PHASE2_FINAL_REPORT.md` §9. On independent data the global/hybrid conclusions
were confirmed: PIRD is the best branch for per-ID attacks, the global branch has
a severe normal-distribution shift (FPR 0.59–0.89 on several captures), and the
hybrid does not recover it. No parameter was tuned on ROAD.

This note documents the interface and the frozen rules.

## What was run

```
python -m experiments.runners.run_phase2 \
    --config experiments/configs/phase2_road_external.yaml --mode road
```

Per-capture pre-injection protocol (see `canguard.exp.road_protocol` and
`experiments/runners/phase_b_road.py`): calibration = pre-injection normals;
test = post-injection traffic. One eligible (non-masquerade, labelled) capture
per attack type, full captures. Frozen global feature definitions,
`window_frames=200`, robust normalization, Isolation Forest parameters, hybrid
rules / λ grid, threshold protocol and adaptive attack parameters.

## Interface notes

* `experiments/runners/phase2_road.py::_prepare_road_hybrid_dataset` adapts the
  frozen pipeline to one ROAD capture and yields the same branch schema used by
  `phase2_hybrid.py`.
* `build_global_features` is dataset-agnostic; only the split/frame loading is
  ROAD-specific.
