## CAN Bus Diagnostic & Anomaly Detection Tool

CAN bus intrusion detection system using the HCRL Car Hacking Dataset (DoS, fuzzing, RPM spoofing, gear spoofing attacks from a Hyundai YF Sonata).

### Datasets

- **HCRL Car-Hacking/** — raw CAN traces (4 attack types + normal baseline)
- Each CSV: `timestamp, CAN_ID, DLC, data[0..7], label`

### Project Structure

| Path | Description |
|------|-------------|
| `eda_hcrl.ipynb` | Exploratory data analysis (8 sections: class balance, ID analysis, payload bytes, DLC, IAT, naive rule baseline) |
| `.venv/` | Python virtual environment (pandas, matplotlib, seaborn, scikit-learn, jupyter) |

### Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter notebook eda_hcrl.ipynb
```

### Dataset Limitations

The HCRL dataset is **presence-based** — every attack is trivially detectable by checking CAN ID novelty or constant payload bytes. ML models scoring near-perfect F1 on this data may not generalize to behavioral attacks that use legitimate IDs with malicious signal values.

Recommended harder benchmarks: ROAD (ORNL), can-train-and-test.
