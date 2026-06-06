from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import ee


@dataclass
class ExportTaskSpec:
    name: str
    description: str
    scale: int
    file_name_prefix: str


def initialize_ee(project_id: str | None = None) -> None:
    project = project_id or os.environ.get("EARTHENGINE_PROJECT")
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        try:
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
        except Exception as exc:
            raise RuntimeError(
                "Earth Engine authentication succeeded, but no Google Cloud project is configured. "
                "Set gee.project_id in config/config.yaml or run: earthengine set_project YOUR_PROJECT_ID"
            ) from exc


def aoi_geometry(cfg: dict[str, Any]) -> ee.Geometry:
    bbox = cfg["aoi"]["bbox"]
    return ee.Geometry.Rectangle(bbox, proj=cfg["aoi"].get("crs", "EPSG:4326"), geodesic=False)


def mask_s2_sr(image: ee.Image, excluded_scl_classes: list[int]) -> ee.Image:
    scl = image.select("SCL")
    keep = ee.Image(1)
    for cls in excluded_scl_classes:
        keep = keep.And(scl.neq(cls))
    return image.updateMask(keep)


def sentinel2_composite(cfg: dict[str, Any], aoi: ee.Geometry) -> ee.Image:
    gee = cfg["gee"]
    cloud = gee["cloud"]
    bands = gee["bands"]
    collection = (
        ee.ImageCollection(gee["sentinel_collection"])
        .filterBounds(aoi)
        .filterDate(gee["start_date"], gee["end_date"])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud["max_scene_cloud_percent"]))
        .map(lambda img: mask_s2_sr(img, cloud["scl_excluded_classes"]))
    )
    return collection.median().select(bands).multiply(0.0001).toFloat().clip(aoi)


def esa_worldcover(cfg: dict[str, Any], aoi: ee.Geometry) -> ee.Image:
    asset = cfg["gee"]["esa_worldcover_asset"]
    return ee.ImageCollection(asset).first().select("Map").toUint8().clip(aoi)


def jrc_gsw_occurrence(cfg: dict[str, Any], aoi: ee.Geometry) -> ee.Image:
    asset = cfg["gee"]["jrc_gsw_asset"]
    return ee.Image(asset).select("occurrence").toFloat().clip(aoi)


def worldpop_population(cfg: dict[str, Any], aoi: ee.Geometry) -> ee.Image:
    gee = cfg["gee"]
    collection = (
        ee.ImageCollection(gee["worldpop_asset"])
        .filterBounds(aoi)
        .filter(ee.Filter.eq("country", gee.get("worldpop_country", "IDN")))
        .filter(ee.Filter.eq("year", int(gee.get("worldpop_year", 2020))))
    )
    return collection.mosaic().select("population").toFloat().clip(aoi)


def export_to_drive(
    image: ee.Image,
    cfg: dict[str, Any],
    spec: ExportTaskSpec,
    region: ee.Geometry,
) -> ee.batch.Task:
    export_cfg = cfg["gee"]["export"]
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=spec.description,
        folder=cfg["drive"]["folder"],
        fileNamePrefix=spec.file_name_prefix,
        region=region,
        scale=spec.scale,
        crs=cfg["aoi"].get("crs", "EPSG:4326"),
        fileDimensions=export_cfg.get("file_dimensions"),
        skipEmptyTiles=True,
        maxPixels=export_cfg["max_pixels"],
    )
    task.start()
    return task


def submit_week1_exports(cfg: dict[str, Any]) -> list[ee.batch.Task]:
    initialize_ee(cfg["gee"].get("project_id"))
    aoi = aoi_geometry(cfg)
    export_cfg = cfg["gee"]["export"]
    aoi_name = cfg["aoi"]["name"]

    specs = [
        (sentinel2_composite(cfg, aoi), ExportTaskSpec("s2", f"s2_composite_{aoi_name}", export_cfg["s2_scale_m"], f"s2_composite_{aoi_name}")),
        (esa_worldcover(cfg, aoi), ExportTaskSpec("esa", f"esa_worldcover_{aoi_name}", export_cfg["label_scale_m"], f"esa_worldcover_{aoi_name}")),
        (jrc_gsw_occurrence(cfg, aoi), ExportTaskSpec("jrc", f"jrc_gsw_occurrence_{aoi_name}", export_cfg["gsw_scale_m"], f"jrc_gsw_occurrence_{aoi_name}")),
        (worldpop_population(cfg, aoi), ExportTaskSpec("worldpop", f"worldpop_population_density_{aoi_name}", export_cfg["gsw_scale_m"], f"worldpop_population_density_{aoi_name}")),
    ]
    return [export_to_drive(image, cfg, spec, aoi) for image, spec in specs]
