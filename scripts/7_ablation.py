from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.ablation import run_ablation
from geoai_flood.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--baseline-fsi", default=None)
    parser.add_argument("--proposed-fsi", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = run_ablation(cfg, args.baseline_fsi, args.proposed_fsi)
    print(f"Ablation results: {out}")


if __name__ == "__main__":
    main()
