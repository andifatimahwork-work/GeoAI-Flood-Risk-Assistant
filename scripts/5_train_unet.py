from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config
from geoai_flood.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    best = train_from_config(cfg, resume=args.resume)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
