from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config
from geoai_flood.gee_export import ExportTaskSpec, aoi_geometry, export_to_drive, initialize_ee, worldpop_population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    initialize_ee(cfg["gee"].get("project_id"))
    aoi = aoi_geometry(cfg)
    aoi_name = cfg["aoi"]["name"]
    task = export_to_drive(
        worldpop_population(cfg, aoi),
        cfg,
        ExportTaskSpec(
            name="worldpop",
            description=f"worldpop_population_density_{aoi_name}",
            scale=cfg["gee"]["export"]["gsw_scale_m"],
            file_name_prefix=f"worldpop_population_density_{aoi_name}",
        ),
        aoi,
    )
    print(f"Started WorldPop export: {task.id}")


if __name__ == "__main__":
    main()
