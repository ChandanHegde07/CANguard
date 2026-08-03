"""Tag existing tables with protocol_version/phase; build ROAD manuscript summary.

Fails loudly if ROAD source is unversioned or not pre_injection_v1.

Usage:
    python experiments/scripts/tag_and_build_manuscript_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from canguard.exp.manuscript_tables import write_road_manuscript_summary
from canguard.exp.metadata import (
    PHASE_A,
    PHASE_B,
    PHASE_C,
    PHASE_WORKSHOP,
    PROTOCOL_HCRL_TEMPORAL,
    PROTOCOL_ROAD_PRE_INJECTION,
    assert_road_protocol_ok,
    ensure_protocol_column,
    tag_dataframe,
)


def _tag_if_exists(path: Path, protocol: str, phase: str, **extra) -> None:
    if not path.exists():
        print(f"skip missing {path}")
        return
    df = pd.read_csv(path)
    df = ensure_protocol_column(df)
    if "protocol" in df.columns and "protocol_version" not in df.columns:
        df["protocol_version"] = df["protocol"]
    # If already tagged with wrong protocol for ROAD files, assert later
    if "protocol_version" not in df.columns:
        df = tag_dataframe(df, protocol_version=protocol, phase=phase, extra=extra or None)
    else:
        # fill missing
        if "phase" not in df.columns:
            df["phase"] = phase
        df["protocol_version"] = df["protocol_version"].fillna(protocol)
        df["phase"] = df["phase"].fillna(phase)
        for k, v in extra.items():
            if k not in df.columns:
                df[k] = v
    df.to_csv(path, index=False)
    print(f"tagged {path} protocol={protocol} phase={phase}")


def main() -> None:
    tables = _ROOT / "tables"
    paper_tables = _ROOT / "paper" / "updated_tables"
    paper_tables.mkdir(parents=True, exist_ok=True)

    # ROAD Phase B (must be corrected protocol)
    road = tables / "road_representation_results.csv"
    if road.exists():
        df = ensure_protocol_column(pd.read_csv(road))
        if "protocol_version" not in df.columns and "protocol" in df.columns:
            df["protocol_version"] = df["protocol"]
        if "protocol_version" not in df.columns:
            # Only auto-tag if Phase B runner left protocol=pre_injection_v1
            raise RuntimeError(
                f"{road} has no protocol metadata — refuse silent tag of possibly stale numbers"
            )
        assert_road_protocol_ok(df)
        # n_captures per attack_type for convenience on row-level file
        if "capture" in df.columns and "attack_type" in df.columns:
            nmap = df.groupby("attack_type")["capture"].nunique().to_dict()
            df["n_captures"] = df["attack_type"].map(nmap)
        df["phase"] = df.get("phase", PHASE_B)
        if "phase" not in df.columns or df["phase"].isna().all():
            df["phase"] = PHASE_B
        else:
            df["phase"] = df["phase"].fillna(PHASE_B)
        df["protocol_version"] = df["protocol_version"].astype(str)
        df.to_csv(road, index=False)
        print(f"verified ROAD results: {road}")

        summary = write_road_manuscript_summary(
            road,
            tables / "road_manuscript_summary.csv",
            representation="residual",
        )
        summary.to_csv(paper_tables / "road_manuscript_summary.csv", index=False)
        print(f"wrote manuscript summary n_rows={len(summary)}")

    for name, proto, phase in [
        ("road_baseline_results.csv", PROTOCOL_ROAD_PRE_INJECTION, PHASE_B),
        ("road_representation_delta.csv", PROTOCOL_ROAD_PRE_INJECTION, PHASE_B),
        ("baseline_results.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_A),
        ("representation_results.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_A),
        ("multiseed_results.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_C),  # mixed corpora
        ("confidence_intervals.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_C),
        ("runtime.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_C),
        ("latency.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_C),
        ("feature_importance.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_C),
        ("feature_ablation.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_C),
        ("statistical_tests.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_C),
        ("error_analysis.csv", PROTOCOL_HCRL_TEMPORAL, PHASE_C),
    ]:
        p = tables / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if name.startswith("road_"):
            df = ensure_protocol_column(df)
            if "protocol_version" not in df.columns and "protocol" in df.columns:
                df["protocol_version"] = df["protocol"]
            if "protocol_version" not in df.columns:
                df["protocol_version"] = proto
            assert_road_protocol_ok(df)
            if "capture" in df.columns and "attack_type" in df.columns:
                nmap = df.groupby("attack_type")["capture"].nunique().to_dict()
                df["n_captures"] = df["attack_type"].map(nmap)
        else:
            if "protocol_version" not in df.columns:
                df["protocol_version"] = proto
            # multiseed mixes HCRL+ROAD — tag more carefully
            if name == "multiseed_results.csv" and "corpus" in df.columns:
                df.loc[df["corpus"] == "road", "protocol_version"] = PROTOCOL_ROAD_PRE_INJECTION
                df.loc[df["corpus"] == "hcrl", "protocol_version"] = PROTOCOL_HCRL_TEMPORAL
            if name == "confidence_intervals.csv" and "corpus" in df.columns:
                df.loc[df["corpus"] == "road", "protocol_version"] = PROTOCOL_ROAD_PRE_INJECTION
                df.loc[df["corpus"] == "hcrl", "protocol_version"] = PROTOCOL_HCRL_TEMPORAL
        if "phase" not in df.columns:
            df["phase"] = phase
        df.to_csv(p, index=False)
        print(f"updated {p.name}")

    # Workshop stamp
    stamp = tables / "workshop_metadata.json"
    import json

    stamp.write_text(
        json.dumps(
            {
                "phase": PHASE_WORKSHOP,
                "road_protocol_required": PROTOCOL_ROAD_PRE_INJECTION,
                "manuscript_road_summary": "tables/road_manuscript_summary.csv",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("done")


if __name__ == "__main__":
    main()
