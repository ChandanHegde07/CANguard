## CAN Bus Diagnostic & Anomaly Detection Tool (CANguard)

CAN bus intrusion detection system using the HCRL Car Hacking Dataset (DoS, fuzzing, RPM spoofing, gear spoofing attacks from a Hyundai YF Sonata).

### Architecture

```
HCRL Car-Hacking CSVs (DoS, Fuzzy, RPM, Gear)
  │
  ├── eda_hcrl.ipynb
  │     DLC-aware CSV loader → class balance → CAN ID analysis
  │     → payload bytes → DLC → IAT → naive rule baseline
  │     → presence-based dataset flag
  │
  ├── feature_eng_hcrl.ipynb
  │     DLC-aware CSV loader → FeaturePipeline (PerIDWindow +
  │     GlobalCANContext) → per-ID sliding windows →
  │     BEHAVIORAL_FEATURES (14 cols) → separability
  │     analysis → Random Forest baseline
  │
  └── pird_hcrl.ipynb
        FeaturePipeline → temporal split (40/20/40) →
        per-ID residual transform (z-score per ID) →
        Isolation Forest (normal-only train) →
        ┌─ residual IF evaluation (RPM, DoS, Fuzzy, Gear)
        ├─ ablation: raw IF vs residual IF
        ├─ supervised HGB reference
        ├─ cross-attack matrix (4x4)
        └─ diagnostic plots (scores, ROC/PR, heatmaps)
```

### Datasets

- **HCRL Car-Hacking/** — raw CAN traces (4 attack types + normal baseline)
- Each CSV: `timestamp, CAN_ID, DLC, data[0..7], label`

Rows with DLC < 8 have fewer data fields on disk. All notebooks use a DLC-aware parser that extracts exactly `dlc` bytes and pads the remainder with NaN, preventing label misalignment.

### Project Structure

| Path | Description |
|------|-------------|
| `eda_hcrl.ipynb` | Exploratory data analysis (8 sections: class balance, ID analysis, payload bytes, DLC, IAT, naive rule baseline) |
| `feature_eng_hcrl.ipynb` | Feature engineering — per-ID sliding-window pipeline extracting IAT stats, byte entropy/variance/max-change, DLC consistency, time-since-last-seen, and ID novelty flags |
| `pird_hcrl.ipynb` | **PIRD (Per-ID Behavioral Residual Detector)** — unsupervised anomaly detection pipeline: per-ID z-score residual transform + Isolation Forest. Includes supervised reference (HistGradientBoosting), cross-attack evaluation, and ablation studies with 5 diagnostic plots |
| `.venv/` | Python virtual environment (pandas, matplotlib, seaborn, scikit-learn, jupyter) |

### Setup

```bash
python3 -m venv .venv
.venv/bin/pip install pandas matplotlib seaborn scikit-learn jupyter nbformat nbconvert
.venv/bin/jupyter notebook
```

### Notebooks

#### 1. EDA (`eda_hcrl.ipynb`)
Class balance, CAN ID analysis, payload byte variance, DLC distribution, inter-arrival time analysis, and a naive rule baseline confirming all 4 attacks are presence-based (F1 > 0.99).

#### 2. Feature Engineering (`feature_eng_hcrl.ipynb`)
Per-ID sliding-window pipeline computing behavioral features. `BEHAVIORAL_FEATURES` list (14 features) excludes presence-based shortcuts. Feature separability analysis with Cohen's d heatmap. Random Forest baseline.

| Feature | Type | Generalizable? |
|---------|------|----------------|
| `iat_mean`, `iat_std`, `iat_median`, `iat_min`, `iat_max` | Temporal — injection cadence | Behavioral |
| `time_since_last_seen` | Temporal — message gap | Behavioral |
| `byte_mean`, `byte_var`, `byte_max_change`, `byte_nunique`, `byte_entropy` | Payload — signal discontinuity | Behavioral |
| `dlc_std`, `dlc_mode` | Protocol — DLC consistency | Behavioral |
| `window_fill` | Window — fill ratio | Behavioral |
| `is_new_id`, `active_ids_1s` | Presence — ID novelty | Presence-based (drop for general models) |

#### 3. PIRD Detector (`pird_hcrl.ipynb`)
Unsupervised detection pipeline with the following stages:

1. **Feature extraction**: Per-ID rolling windows → 14 behavioral features
2. **Temporal split**: 40% calib / 20% train / 40% test (no shuffle)
3. **Per-ID residual transform**: z-score normalization per CAN ID using calib normal mean/std; unseen IDs fall back to global normal stats
4. **Isolation Forest**: Trained on normal-only train residuals; threshold at ~1% FPR on held-out normal validation
5. **Evaluation**: Precision, recall, F1, ROC-AUC, PR-AUC per dataset
6. **Ablation**: Raw behavioral IF vs residual IF
7. **Supervised reference**: HistGradientBoosting on residuals
8. **Cross-attack**: 4x4 recall/FPR matrix (train on source, test on target with target's per-ID stats)
9. **Plots**: Score distributions, operating point summary, ablation comparison, cross-attack heatmaps, RPM ROC/PR curves

### Key Findings

- All 4 HCRL attacks are **presence-based**: naive rules (new ID, dominant ID, constant byte) achieve F1 > 0.99
- Behavioral features (IAT, byte change, time-since-last-seen) separate normal vs attack, but are less dominant
- PIRD (per-ID residuals + Isolation Forest) uses behavioral features only, with no exposure to presence-based shortcuts
- **Dataset limitation**: near-perfect scores on HCRL do not prove generalization to stealth attacks that reuse legitimate CAN IDs

### Recommended Harder Benchmarks

- ROAD (Real ORNL Automotive Dynamics) — attacks injected over legitimate IDs
- can-train-and-test — synthetic CAN attacks with configurable stealth level
