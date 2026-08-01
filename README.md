# CANguard

Behavioral CAN Bus Intrusion Detection System using the HCRL Car-Hacking Dataset (DoS, fuzzing, RPM spoofing, gear spoofing — Hyundai YF Sonata).

## System Architecture

```mermaid
flowchart TD
    subgraph Data
        CSV[("HCRL Car-Hacking CSVs")] --> Loader["canguard.data<br/>HCRLLoader + factory"]
        Loader --> Schema["Canonical schema<br/>timestamp, can_id, dlc, data_0..7, label"]
    end

    subgraph Features
        Schema --> Win["canguard.features<br/>PerIDWindow sliding window"]
        Win --> BFeats["14 BEHAVIORAL_FEATURES<br/>IAT • DLC • payload • misc"]
        BFeats --> Residual["per-ID z-score residual<br/>fit_per_id_stats → transform_residuals"]
        Residual --> Split["temporal split<br/>40% calib / 20% train / 40% test"]
    end

    subgraph Detection
        Split --> IF["canguard.detectors<br/>IsolationForestDetector<br/>(normal-only, 200 trees)"]
    end

    subgraph Evaluation
        IF --> Metrics["canguard.evaluation<br/>precision • recall • F1 • ROC • PR"]
        IF --> Cross["cross-attack matrix<br/>sweep thresholds • ablation"]
        Metrics --> Viz["canguard.visualization"]
        Viz --> Figs["figures/ (.png)"]
        Metrics --> Res["results/ (.json)"]
    end
```

**Pipeline**: per-ID sliding windows → 14 behavioral features → z-score residuals per ID → Isolation Forest (normal-only train) → metrics / figures / results.

## Data Flow

```mermaid
sequenceDiagram
    participant CSV as Raw CSV
    participant L as HCRLLoader
    participant F as FeaturePipeline
    participant R as ResidualTransform
    participant D as IsolationForest
    participant E as Evaluation

    CSV->>L: rows (DLC-aware parse)
    L->>F: canonical DataFrame
    F->>F: per-ID sliding windows (14 features)
    F->>R: feature windows
    R->>R: per-ID z-score residualization
    R->>D: residual vectors (train-only normals)
    D->>E: anomaly scores + threshold
    E->>E: metrics, cross-attack, figures, results
```

## Project Structure

| Path | Description |
|------|-------------|
| `src/canguard/` | Library: `data`, `features`, `detectors`, `evaluation`, `visualization` |
| `notebooks/` | Research notebooks: `eda_hcrl`, `feature_eng_hcrl`, `pird_hcrl`, `pird_v2_extensions` |
| `experiments/` | Config-driven CLI runners (`--config *.yaml`) |
| `tests/` | `pytest` suite (45 tests) |
| `configs/` | Experiment config files |
| `run_pipeline.py` | One-command full pipeline runner → `results/` + `figures/` |
| `figures/` | Generated diagnostic plots (`.png`) |
| `results/` | Machine-readable metrics (`.json`) |
| `HCRL Car-Hacking/` | Raw dataset CSVs (git-ignored) |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Usage

```bash
# Run the full pipeline and generate figures + results
.venv/bin/python run_pipeline.py

# Or run a single experiment from a config
.venv/bin/python -m experiments.runners.train_detector --config experiments/configs/hcrl.yaml
```

## Key Results (residual IF @ ~1% FPR)

| Dataset | F1 | Recall | FPR |
|---------|-----|--------|-----|
| DoS | 0.017 | 0.01 | 0.055 |
| Fuzzy | 0.469 | 0.98 | 0.147 |
| RPM | 0.991 | 0.999 | 0.004 |
| gear | 0.945 | 0.998 | 0.032 |

Per-ID residuals work well on RPM/gear but **fail on DoS** (novel-ID flooding not captured by per-ID statistics). Supervised reference (HGB) reaches F1 = 1.0 on DoS/RPM/gear.

## Key Findings

- All 4 HCRL attacks are **presence-based** — naive rules (new ID, dominant ID, constant byte) achieve F1 > 0.99
- Behavioral features separate normal vs attack but do not generalize trivially
- **Dataset limitation**: near-perfect scores on HCRL do not prove generalization to stealth attacks that reuse legitimate IDs

## Recommended Harder Benchmarks

- [ROAD](https://www.nist.gov/programs-projects/road-real-ornl-automotive-drivermodel-data) — attacks injected over legitimate IDs
- can-train-and-test — synthetic CAN attacks with configurable stealth level

## License

This project is licensed under the [Apache License 2.0](LICENSE).
