"""CLI runner: feature-group ablation from a YAML config.

Usage:
    python -m experiments.runners.run_ablation --config experiments/configs/ablation.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .train_detector import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature-group ablation runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    base = dict(config)
    base.pop("variants", None)

    aggregated: dict = {}
    for variant in config["variants"]:
        vc = dict(base)
        vc["name"] = variant["name"]
        # feature_groups resolves the selected subsets; exclude_groups left
        # for a future exclusion-aware feature resolver.
        vc["variant_feature_groups"] = variant.get("feature_groups")
        results = run_experiment(vc)
        aggregated[variant["name"]] = results
        print(f"\n=== variant: {variant['name']} ===")
        for name, m in results.items():
            print(f"  {name}: F1={m['f1']:.3f} Recall={m['recall']:.3f}")

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_results.json"
    out_path.write_text(json.dumps(aggregated, indent=2, default=str))
    print(f"\nWrote aggregate ablation results to {out_path}")


if __name__ == "__main__":
    main()
