## CAN Bus Diagnostic & Anomaly Detection Tool (CANguard)

CAN bus intrusion detection system using the HCRL Car Hacking Dataset (DoS, fuzzing, RPM spoofing, gear spoofing attacks from a Hyundai YF Sonata).

### Datasets

- **HCRL Car-Hacking/** — raw CAN traces (4 attack types + normal baseline)
- Each CSV: `timestamp, CAN_ID, DLC, data[0..7], label`

Rows with DLC < 8 have fewer data fields on disk. The loader in both notebooks uses a DLC-aware parser that extracts exactly `dlc` bytes and pads the remainder with NaN, preventing label misalignment.

### Project Structure

| Path | Description |
|------|-------------|
| `eda_hcrl.ipynb` | Exploratory data analysis (8 sections: class balance, ID analysis, payload bytes, DLC, IAT, naive rule baseline) |
| `feature_eng_hcrl.ipynb` | Feature engineering — per-ID sliding-window pipeline extracting IAT stats, byte entropy/variance/max-change, DLC consistency, time-since-last-seen, and ID novelty flags |
| `.venv/` | Python virtual environment (pandas, matplotlib, seaborn, scikit-learn, jupyter) |

### Setup

```bash
python3 -m venv .venv
.venv/bin/pip install pandas matplotlib seaborn scikit-learn jupyter nbformat nbconvert
.venv/bin/jupyter notebook eda_hcrl.ipynb
```

### Feature Engineering

The feature pipeline in `feature_eng_hcrl.ipynb` computes behavioral features over a per-ID rolling window:

| Feature | Type | Generalizable? |
|---------|------|----------------|
| `iat_mean`, `iat_std`, `iat_median` | Temporal — injection cadence | Behavioral |
| `time_since_last_seen` | Temporal — message gap | Behavioral |
| `byte_max_change`, `byte_var`, `byte_entropy` | Payload — signal discontinuity | Behavioral |
| `dlc_std`, `dlc_mode` | Protocol — DLC consistency | Behavioral |
| `is_new_id`, `active_ids_1s` | Presence — ID novelty | Presence-based (drop for general models) |

### Key Findings

- All 4 HCRL attacks are **presence-based**: naive rules (new ID, dominant ID, constant byte) achieve F1 > 0.99
- Behavioral features (IAT, byte change, time-since-last-seen) still separate normal vs attack, but are less dominant than the trivial ID-based features
- **Dataset limitation**: ML models trained on HCRL with `is_new_id` will appear near-perfect but fail on stealth attacks that reuse legitimate CAN IDs

### Recommended Harder Benchmarks

- ROAD (Real ORNL Automotive Dynamics) — attacks injected over legitimate IDs
- can-train-and-test — synthetic CAN attacks with configurable stealth level
