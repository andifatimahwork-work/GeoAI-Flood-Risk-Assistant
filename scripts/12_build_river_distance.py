from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoai_flood.config import load_config, resolve_path


def waterways_from_osm(cfg: dict) -> gpd.GeoDataFrame:
    west, south, east, north = cfg["aoi"]["bbox"]
    tags = cfg.get("river_distance", {}).get("osm_tags", {"waterway": True})
    # OSMnx 2.x expects bbox=(left, bottom, right, top).
    gdf = ox.features_from_bbox((west, south, east, north), tags=tags)
    if gdf.empty:
        raise RuntimeError("No OSM waterway features found in AOI.")
    gdf = gdf.reset_index()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise RuntimeError("OSM waterway query returned no line/polygon geometries.")
    return gdf.to_crs("EPSG:4326")


def build_river_distance(cfg: dict) -> Path:
    root = Path(cfg["_project_root"])
    ref_path = resolve_path(cfg["fsi"]["reference_tif"], root)
    out_path = resolve_path(cfg["fsi"]["river_distance_tif"], root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    waterways = waterways_from_osm(cfg)
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        shape = (ref.height, ref.width)

    river_mask = rasterize(
        [(geom, 1) for geom in waterways.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    pixel_width_m = 111_320.0 * abs(transform.a)
    pixel_height_m = 110_540.0 * abs(transform.e)
    distance = distance_transform_edt(river_mask == 0, sampling=(pixel_height_m, pixel_width_m)).astype(np.float32)
    max_distance = float(cfg.get("river_distance", {}).get("max_distance_m", 10000))
    distance = np.clip(distance, 0.0, max_distance)

    profile.update(count=1, dtype="float32", nodata=None, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(distance, 1)
    waterways.to_file(out_path.with_suffix(".geojson"), driver="GeoJSON")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = build_river_distance(cfg)
    print(f"River distance raster saved: {out}")


if __name__ == "__main__":
    main()
