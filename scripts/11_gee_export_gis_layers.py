from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config
from geoai_flood.gee_export import ExportTaskSpec, aoi_geometry, export_to_drive, initialize_ee


def srtm_elevation(cfg: dict, aoi: ee.Geometry) -> ee.Image:
    return ee.Image(cfg["gee"]["srtm_asset"]).select("elevation").toFloat().clip(aoi)


def srtm_slope(cfg: dict, aoi: ee.Geometry) -> ee.Image:
    elevation = ee.Image(cfg["gee"]["srtm_asset"]).select("elevation")
    return ee.Terrain.slope(elevation).rename("slope").toFloat().clip(aoi)


def chirps_mean_annual(cfg: dict, aoi: ee.Geometry) -> ee.Image:
    start_year = int(cfg["gee"]["chirps_start_year"])
    end_year = int(cfg["gee"]["chirps_end_year"])
    start = f"{start_year}-01-01"
    end = f"{end_year + 1}-01-01"
    years = end_year - start_year + 1
    total = (
        ee.ImageCollection(cfg["gee"]["chirps_daily_asset"])
        .filterDate(start, end)
        .filterBounds(aoi)
        .select("precipitation")
        .sum()
    )
    return total.divide(years).rename("rainfall").toFloat().clip(aoi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    initialize_ee(cfg["gee"].get("project_id"))
    aoi = aoi_geometry(cfg)
    aoi_name = cfg["aoi"]["name"]
    scale = int(cfg["gee"]["export"]["gsw_scale_m"])

    exports = [
        (
            srtm_elevation(cfg, aoi),
            ExportTaskSpec("elevation", f"elevation_srtm_30m_{aoi_name}", scale, f"elevation_srtm_30m_{aoi_name}"),
        ),
        (
            srtm_slope(cfg, aoi),
            ExportTaskSpec("slope", f"slope_srtm_30m_{aoi_name}", scale, f"slope_srtm_30m_{aoi_name}"),
        ),
        (
            chirps_mean_annual(cfg, aoi),
            ExportTaskSpec(
                "chirps",
                f"chirps_mean_annual_{cfg['gee']['chirps_start_year']}_{cfg['gee']['chirps_end_year']}_{aoi_name}",
                scale,
                f"chirps_mean_annual_{cfg['gee']['chirps_start_year']}_{cfg['gee']['chirps_end_year']}_{aoi_name}",
            ),
        ),
    ]
    for image, spec in exports:
        task = export_to_drive(image, cfg, spec, aoi)
        print(f"Started {spec.name} export: {task.id}")


if __name__ == "__main__":
    main()
