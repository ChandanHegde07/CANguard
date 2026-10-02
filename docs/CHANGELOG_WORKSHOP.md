# CHANGELOG — Workshop readiness pass

**Date:** 2026-08-03  
**Scope:** Code-side workshop readiness (Tasks 1–5). No residualization math changes. No negative results removed.

---

## Why this pass

`FINAL_REPORT_PHASE_C.md` assessed the project as arXiv-ready but workshop-needing: corrected ROAD numbers in the manuscript path, one external baseline, n=1 std hygiene, IEEE figure consistency, and protocol versioning.

---

## Tables changed / added

| Path | Change |
|------|--------|
| `tables/road_representation_results.csv` | Verified `protocol`/`protocol_version` = `pre_injection_v1`; added `n_captures`, `phase` |
| `tables/road_baseline_results.csv` | Same protocol verification + metadata |
| `tables/road_representation_delta.csv` | Protocol verification + metadata |
| `tables/road_manuscript_summary.csv` | **New** — attack×detector residual summary with `n_captures` and `*_fmt` (suppresses ±std when n=1) |
| `paper/updated_tables/road_manuscript_summary.csv` | Copy for manuscript authors |
| `tables/external_baseline_results.csv` | **New** — `sequence_autoencoder` on HCRL+ROAD, raw+residual, same protocol |
| `tables/workshop_metadata.json` | **New** — required ROAD protocol stamp |
| Other Phase A/C CSVs | Added `protocol_version` / `phase` where missing (does not recompute metrics) |

### ROAD protocol assertion

Manuscript aggregation **raises** (does not warn) if:

- `protocol_version` / `protocol` is missing, or  
- value is not `pre_injection_v1` (or documented frame-selection variant)

Source of truth for ROAD paper numbers: **`tables/road_representation_results.csv`** (Phase B corrected protocol), summarized by `tables/road_manuscript_summary.csv`.  
Do **not** paste pre-fix notebook / old Table VI numbers.

### External baseline (not SOTA)

| Detector key | `sequence_autoencoder` |
|--------------|------------------------|
| Role | Comparison point (CANet / sequence reconstruction–inspired) |
| Input | **Same 14-d window features** as other detectors (raw or residual sequences of length 5), **not** raw CAN bytes |
| Implementation | CPU MLP-AE over flattened short sequences (`src/canguard/detectors/sequence_autoencoder.py`) |
| Runner | `python -m experiments.runners.run_external_baseline --config experiments/configs/external_baseline.yaml` |

**Example residual F1 (kept as-is; not optimized):** HCRL RPM 0.54, gear 0.57, DoS 0.23; ROAD reverse_light residual ~0.22–0.28 — generally **weaker** than residual IF/HBOS on spoofing tasks. Negative/weak cells retained.

---

## Code added / updated

| Component | Path |
|-----------|------|
| Protocol metadata | `src/canguard/exp/metadata.py` |
| Manuscript formatters + ROAD summary | `src/canguard/exp/manuscript_tables.py` |
| Sequence AE detector | `src/canguard/detectors/sequence_autoencoder.py` |
| Registry entry | `src/canguard/detectors/registry.py` |
| External baseline runner | `experiments/runners/run_external_baseline.py` |
| Tag + manuscript builder | `experiments/scripts/tag_and_build_manuscript_tables.py` |
| IEEE figure regen | `experiments/scripts/regenerate_ieee_figures.py` |
| Shared IEEE style | `src/canguard/visualization/style.py` (colorblind palette, single/double column widths, 300 dpi, Times/serif) |
| Phase A/B tagging | `run_phase_a.py`, `run_phase_b_road.py` write `protocol_version` / `phase` |
| Tests | `tests/test_manuscript_tables.py` |

---

## Figures regenerated (presentation only)

Under `figures/` and `paper/updated_figures/`:

- `multiseed_delta_f1_{hcrl,road}.png`
- `perm_importance_*.png`
- `runtime_train_score.png`, `runtime_throughput.png`
- `latency_hist.png`
- `coverage_boundary.png`

No underlying metric values were changed—style only.

---

## n=1 std rule

`format_mean_std(mean, std, n)`:

- `n < 2` → e.g. `0.533 (n=1; std n/a)` — **no** `±`  
- `n ≥ 2` → `0.533 ± 0.012 (n=3)`

Unit tests: `test_n1_captures_suppress_std`, `test_road_summary_n_captures_and_fmt`.

---

## Commands to reproduce this pass

```bash
pip install -e ".[dev]"
python experiments/scripts/tag_and_build_manuscript_tables.py
python -m experiments.runners.run_external_baseline --config experiments/configs/external_baseline.yaml
python experiments/scripts/regenerate_ieee_figures.py
python -m pytest tests/ -q
```

---

## What authors should do next (prose, out of this pass)

1. Replace any ROAD table in the TeX draft with numbers from `tables/road_manuscript_summary.csv` (or residual columns of `road_representation_results.csv`).  
2. Add one Related Work / Results paragraph citing `sequence_autoencoder` as a lightweight sequence-reconstruction **baseline**, not a competitor we claim to beat comprehensively.  
3. When reporting ROAD coolant / single-capture attacks, use the `*_fmt` column so n=1 never shows `±0.000`.

---

## Explicit non-changes

- Residual z-score math  
- FPR-target thresholding  
- Existing detector interfaces  
- Retention of DoS / fuzzing / AE-OCSVM negative findings  
