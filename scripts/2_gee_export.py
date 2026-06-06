from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config
from geoai_flood.gee_export import submit_week1_exports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tasks = submit_week1_exports(cfg)
    for task in tasks:
        print(f"Started GEE export: {task.id}")


if __name__ == "__main__":
    main()
