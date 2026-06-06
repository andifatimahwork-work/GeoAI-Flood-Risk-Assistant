from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config
from geoai_flood.preprocess import make_tiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train, val = make_tiles(cfg)
    print(f"Created {len(train)} train tiles and {len(val)} val tiles.")


if __name__ == "__main__":
    main()
