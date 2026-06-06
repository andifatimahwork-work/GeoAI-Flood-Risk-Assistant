from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config
from geoai_flood.evaluate import evaluate_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = evaluate_from_config(cfg, args.checkpoint)
    print(f"Evaluation outputs: {out_dir}")


if __name__ == "__main__":
    main()
