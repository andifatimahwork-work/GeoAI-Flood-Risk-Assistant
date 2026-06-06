from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer

from .config import load_config, load_json, resolve_path
from .env import load_project_env

load_project_env()


def classify_fsi(score: float | None, cfg: dict[str, Any]) -> str:
    if score is None or not np.isfinite(score):
        return "NoData/Masked"

    categories = cfg.get("fsi", {}).get("categories", {})
    low_max = float(categories.get("low_max", 0.65))
    medium_high_max = float(categories.get("medium_high_max", 0.73))
    high_max = float(categories.get("high_max", 0.76))

    if score < low_max:
        return "Low"
    if score < medium_high_max:
        return "Medium-High"
    if score < high_max:
        return "High"
    return "Very High"


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _sample_raster(path: Path, lat: float, lon: float) -> float | None:
    if not path.exists():
        return None

    with rasterio.open(path) as src:
        x, y = lon, lat
        if src.crs and src.crs.to_string() != "EPSG:4326":
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)

        if not (src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top):
            return None

        row, col = src.index(x, y)
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return None

        value = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
        if src.nodata is not None and value == src.nodata:
            return None
        if not np.isfinite(value):
            return None
        return float(value)


def _lulc_info(class_value: float | None, cfg: dict[str, Any]) -> dict[str, Any]:
    if class_value is None or not np.isfinite(class_value):
        return {"class_id": None, "class_name": None, "score": None}

    class_id = int(round(class_value))
    names = cfg["classes"]["names"]
    class_name = names[class_id] if 0 <= class_id < len(names) else None
    score = None
    if class_name:
        score = float(cfg["fsi"]["lulc_risk_scores"].get(class_name, np.nan))
        if not np.isfinite(score):
            score = None
    return {"class_id": class_id, "class_name": class_name, "score": score}


def _cache_key(lat: float, lon: float, mode: str, config_path: str | Path) -> str:
    payload = {
        "lat": round(float(lat), 4),
        "lon": round(float(lon), 4),
        "mode": mode,
        "config": str(config_path),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    try:
        import diskcache

        cache_dir = os.getenv("CACHE_PATH", "cache")
        cache = diskcache.Cache(cache_dir)
        return cache.get(key)
    except Exception:
        return None


def _cache_set(key: str, value: dict[str, Any], cfg: dict[str, Any]) -> None:
    try:
        import diskcache

        cache_dir = os.getenv("CACHE_PATH", "cache")
        ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
        cache = diskcache.Cache(cache_dir)
        cache.set(key, value, expire=ttl_seconds)
    except Exception:
        return


def _linear_score(value: float | None, lo: float, hi: float, inverse: bool = False) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    scaled = (float(value) - lo) / max(hi - lo, 1e-6)
    scaled = float(np.clip(scaled, 0.0, 1.0))
    return 1.0 - scaled if inverse else scaled


def _elevation_score(value: float | None, max_elev: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    elev = max(float(value), 0.0)
    if max_elev <= 0:
        return 1.0
    return float(np.clip(1.0 - (np.log1p(elev) / np.log1p(max_elev)), 0.0, 1.0))


def _read_valid(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr[np.isfinite(arr)]


@lru_cache(maxsize=8)
def normalization_reference(config_path: str) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])
    fsi_cfg = cfg["fsi"]

    refs: dict[str, Any] = {}
    for key, path_key in [
        ("slope", "slope_tif"),
        ("rainfall", "rainfall_tif"),
        ("river_distance", "river_distance_tif"),
        ("population", "population_tif"),
    ]:
        arr = _read_valid(resolve_path(fsi_cfg[path_key], root))
        refs[key] = {
            "p02": float(np.nanpercentile(arr, 2)),
            "p98": float(np.nanpercentile(arr, 98)),
        }

    elev = _read_valid(resolve_path(fsi_cfg["elevation_tif"], root))
    refs["elevation"] = {"max": float(np.nanmax(np.maximum(elev, 0.0)))}
    return refs


def monte_carlo_ci_point(
    breakdown: dict[str, float | None],
    weights: dict[str, float],
    iterations: int,
    perturbation_fraction: float,
    seed: int,
) -> tuple[float | None, float | None]:
    keys = list(weights.keys())
    if any(breakdown.get(key) is None for key in keys):
        return None, None
    values = np.asarray([float(breakdown[key]) for key in keys], dtype=np.float32)
    base = np.asarray([float(weights[key]) for key in keys], dtype=np.float32)
    rng = np.random.default_rng(seed)
    scores = np.empty(iterations, dtype=np.float32)
    for i in range(iterations):
        noise = rng.uniform(1.0 - perturbation_fraction, 1.0 + perturbation_fraction, size=len(keys))
        sampled = base * noise
        sampled = sampled / sampled.sum()
        scores[i] = float((values * sampled).sum())
    return float(np.nanpercentile(scores, 5)), float(np.nanpercentile(scores, 95))


def weighted_fsi_score(breakdown: dict[str, float | None], cfg: dict[str, Any]) -> float | None:
    weights = {k: float(v) for k, v in cfg["fsi"]["weights"].items()}
    if any(breakdown.get(key) is None for key in weights):
        return None
    score = sum(float(breakdown[key]) * weights[key] for key in weights) / max(sum(weights.values()), 1e-6)
    return float(score)


def _format_result(
    lat: float,
    lon: float,
    cfg: dict[str, Any],
    breakdown: dict[str, float | None],
    lulc: dict[str, Any],
    computation_mode: str,
    raw_values: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    air_masked = lulc.get("class_name") in set(cfg["fsi"].get("mask_lulc_classes", []))
    fsi_score = None if air_masked else weighted_fsi_score(breakdown, cfg)
    ci_low, ci_high = (None, None)
    if fsi_score is not None:
        ci_low, ci_high = monte_carlo_ci_point(
            breakdown,
            {k: float(v) for k, v in cfg["fsi"]["weights"].items()},
            iterations=int(cfg["fsi"].get("monte_carlo_iterations", 1000)),
            perturbation_fraction=float(cfg["fsi"].get("perturbation_fraction", 0.10)),
            seed=int(cfg["project"].get("seed", 42)),
        )

    weights = {k: float(v) for k, v in cfg["fsi"]["weights"].items()}
    weighted_breakdown = {
        key: _round_or_none(value * weights[key], 4) if value is not None and key in weights else None
        for key, value in breakdown.items()
    }
    top_factors = [
        {"factor": key, "weighted_contribution": value}
        for key, value in sorted(
            ((k, v) for k, v in weighted_breakdown.items() if isinstance(v, (int, float))),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
    ]
    return {
        "lat": float(lat),
        "lon": float(lon),
        "fsi_score": _round_or_none(fsi_score, 4),
        "ci_low": _round_or_none(ci_low, 4),
        "ci_high": _round_or_none(ci_high, 4),
        "category": classify_fsi(fsi_score, cfg),
        "threshold": float(cfg["validation"]["fsi_threshold"]),
        "breakdown": {key: _round_or_none(value, 4) for key, value in breakdown.items()},
        "weighted_breakdown": weighted_breakdown,
        "top_dominant_factors": top_factors,
        "lulc": lulc,
        "raw_values": raw_values or {},
        "computation_mode": computation_mode,
        "cached": False,
        "warnings": warnings or [],
        "notes": (
            "On-the-fly FSI uses GEE layers when available. The Week-1 FSI raster remains a validation, "
            "visualization, and credibility artifact."
        ),
    }


def get_flood_score_local(lat: float, lon: float, cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["_project_root"])
    fsi_cfg = cfg["fsi"]
    component_dir = resolve_path(fsi_cfg["output_components_dir"], root)

    fsi_score = _sample_raster(resolve_path(fsi_cfg["output_fsi_tif"], root), lat, lon)
    ci_low = _sample_raster(resolve_path(fsi_cfg["output_ci_low_tif"], root), lat, lon)
    ci_high = _sample_raster(resolve_path(fsi_cfg["output_ci_high_tif"], root), lat, lon)
    lulc_raw = _sample_raster(resolve_path(fsi_cfg["lulc_unet_tif"], root), lat, lon)
    lulc = _lulc_info(lulc_raw, cfg)

    breakdown = {
        "elevation": _sample_raster(component_dir / "elevation_score.tif", lat, lon),
        "slope": _sample_raster(component_dir / "slope_score.tif", lat, lon),
        "rainfall": _sample_raster(component_dir / "rainfall_score.tif", lat, lon),
        "river_distance": _sample_raster(component_dir / "river_distance_score.tif", lat, lon),
        "lulc": lulc["score"],
        "population": _sample_raster(component_dir / "population_score.tif", lat, lon),
    }
    result = _format_result(lat, lon, cfg, breakdown, lulc, "local_raster", warnings=["Local raster fallback was used."])
    result["fsi_score"] = _round_or_none(fsi_score, 4)
    result["ci_low"] = _round_or_none(ci_low, 4)
    result["ci_high"] = _round_or_none(ci_high, 4)
    result["category"] = classify_fsi(fsi_score, cfg)
    result["source"] = {
        "fsi": str(resolve_path(fsi_cfg["output_fsi_tif"], root)),
        "ci_low": str(resolve_path(fsi_cfg["output_ci_low_tif"], root)),
        "ci_high": str(resolve_path(fsi_cfg["output_ci_high_tif"], root)),
        "lulc": str(resolve_path(fsi_cfg["lulc_unet_tif"], root)),
    }
    return result


def initialize_ee_service(cfg: dict[str, Any]):
    import ee

    project = cfg["gee"].get("project_id") or os.getenv("EARTHENGINE_PROJECT")
    key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    if key_path:
        key_file = Path(key_path)
        if not key_file.exists():
            raise FileNotFoundError(f"GEE_SERVICE_ACCOUNT_KEY file not found: {key_file}")
        key = json.loads(key_file.read_text(encoding="utf-8"))
        service_account = key["client_email"]
        credentials = ee.ServiceAccountCredentials(service_account, str(key_file))
        ee.Initialize(credentials, project=project)
        return ee

    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    return ee


def gee_live_status(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    project = cfg["gee"].get("project_id") or os.getenv("EARTHENGINE_PROJECT")
    key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    personal_credentials = Path.home() / ".config" / "earthengine" / "credentials"
    return {
        "earthengine_project_set": bool(project),
        "earthengine_project": project,
        "service_account_key_set": bool(key_path),
        "service_account_key_exists": bool(key_path and Path(key_path).exists()),
        "personal_credentials_exists": personal_credentials.exists(),
        "ready_for_local_live": bool(project or key_path or personal_credentials.exists()),
        "ready_for_service_deployment": bool(project and key_path and Path(key_path).exists()),
        "notes": (
            "Local live mode can use personal Earth Engine auth plus EARTHENGINE_PROJECT. "
            "Service deployment should use GEE_SERVICE_ACCOUNT_KEY and EARTHENGINE_PROJECT."
        ),
    }


def gee_raw_values(lat: float, lon: float, cfg: dict[str, Any]) -> dict[str, Any]:
    ee = initialize_ee_service(cfg)
    point = ee.Geometry.Point([float(lon), float(lat)])
    gee = cfg["gee"]

    elevation = ee.Image(gee["srtm_asset"]).select("elevation").toFloat()
    slope = ee.Terrain.slope(elevation).rename("slope").toFloat()
    years = int(gee["chirps_end_year"]) - int(gee["chirps_start_year"]) + 1
    rainfall = (
        ee.ImageCollection(gee["chirps_daily_asset"])
        .filterDate(f"{gee['chirps_start_year']}-01-01", f"{int(gee['chirps_end_year']) + 1}-01-01")
        .select("precipitation")
        .sum()
        .divide(years)
        .rename("rainfall")
        .toFloat()
    )
    population = (
        ee.ImageCollection(gee["worldpop_asset"])
        .filterBounds(point)
        .filter(ee.Filter.eq("country", gee.get("worldpop_country", "IDN")))
        .filter(ee.Filter.eq("year", int(gee.get("worldpop_year", 2020))))
        .mosaic()
        .select("population")
        .rename("population")
        .toFloat()
    )

    raw = {
        "elevation_m": _reduce_region_first(ee, elevation, point, 30, "elevation"),
        "slope_deg": _reduce_region_first(ee, slope, point, 30, "slope"),
        "rainfall_mm_year": _reduce_region_first(ee, rainfall, point, 5000, "rainfall"),
        "population_density": _reduce_region_first(ee, population, point, 100, "population"),
    }

    river_asset = gee.get("river_fc_asset") or os.getenv("GEE_RIVER_FC_ASSET")
    if river_asset:
        try:
            rivers = ee.FeatureCollection(river_asset)
            distance = rivers.distance(searchRadius=float(cfg["river_distance"].get("max_distance_m", 10000))).rename("river_distance")
            raw["river_distance_m"] = _reduce_region_first(ee, distance, point, 30, "river_distance")
        except Exception as exc:
            raw["river_distance_m"] = None
            raw["river_distance_warning"] = f"GEE river distance failed: {exc}"
    else:
        root = Path(cfg["_project_root"])
        raw["river_distance_m"] = _sample_raster(resolve_path(cfg["fsi"]["river_distance_tif"], root), lat, lon)
        raw["river_distance_warning"] = "River distance sampled from local OSM-derived raster; publish it as a GEE asset for full server-side computation."
    return raw


def _reduce_region_first(ee: Any, image: Any, point: Any, scale: int, band: str) -> float | None:
    value = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=scale,
        bestEffort=True,
    ).get(band).getInfo()
    return None if value is None else float(value)


def _center_crop_or_pad(arr: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.float32)
    h, w = arr.shape
    crop_h = min(h, size)
    crop_w = min(w, size)
    src_r0 = max((h - crop_h) // 2, 0)
    src_c0 = max((w - crop_w) // 2, 0)
    dst_r0 = (size - crop_h) // 2
    dst_c0 = (size - crop_w) // 2
    out[dst_r0 : dst_r0 + crop_h, dst_c0 : dst_c0 + crop_w] = arr[src_r0 : src_r0 + crop_h, src_c0 : src_c0 + crop_w]
    return out


def gee_sentinel_patch(lat: float, lon: float, cfg: dict[str, Any]) -> np.ndarray:
    from .gee_export import mask_s2_sr

    ee = initialize_ee_service(cfg)
    gee = cfg["gee"]
    tile_size = int(cfg["preprocessing"]["tile_size"])
    half_m = tile_size * int(gee["export"]["s2_scale_m"]) / 2
    lat_delta = half_m / 111_320.0
    lon_delta = half_m / (111_320.0 * max(np.cos(np.deg2rad(lat)), 0.2))
    region = ee.Geometry.Rectangle([lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta], geodesic=False)
    collection = (
        ee.ImageCollection(gee["sentinel_collection"])
        .filterBounds(region)
        .filterDate(gee["start_date"], gee["end_date"])
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", gee["cloud"]["max_scene_cloud_percent"]))
        .map(lambda img: mask_s2_sr(img, gee["cloud"]["scl_excluded_classes"]))
    )
    image = (
        collection.median()
        .select(gee["bands"])
        .multiply(0.0001)
        .toFloat()
        .reproject("EPSG:3857", None, int(gee["export"]["s2_scale_m"]))
    )
    sample = image.sampleRectangle(region=region, defaultValue=0).getInfo()["properties"]
    bands = []
    for band in gee["bands"]:
        arr = np.asarray(sample[band], dtype=np.float32)
        bands.append(_center_crop_or_pad(arr, tile_size))
    return np.stack(bands, axis=0)


@lru_cache(maxsize=2)
def _load_model_and_stats(config_path: str):
    import torch

    from .model import build_unet

    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])
    stats = load_json(resolve_path(cfg["paths"]["dataset_stats"], root))
    checkpoint = resolve_path(cfg["training"]["checkpoint_dir"], root) / cfg["training"]["best_model_name"]
    model = build_unet(
        cfg["training"]["encoder"],
        None,
        cfg["training"]["in_channels"],
        cfg["classes"]["num_classes"],
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, stats


def predict_lulc_on_the_fly(lat: float, lon: float, cfg: dict[str, Any], config_path: str) -> dict[str, Any]:
    import torch

    patch = gee_sentinel_patch(lat, lon, cfg)
    model, stats = _load_model_and_stats(str(config_path))
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    image = (np.clip(patch, 0.0, 1.0) - mean) / std
    with torch.no_grad():
        tensor = torch.from_numpy(image).unsqueeze(0)
        pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
    center_row = pred.shape[0] // 2
    center_col = pred.shape[1] // 2
    center = int(pred[center_row, center_col])

    window_size = int(cfg.get("inference", {}).get("lulc_center_window_size", 5))
    window_size = max(1, window_size if window_size % 2 == 1 else window_size + 1)
    half = window_size // 2
    window = pred[
        max(center_row - half, 0) : min(center_row + half + 1, pred.shape[0]),
        max(center_col - half, 0) : min(center_col + half + 1, pred.shape[1]),
    ]
    counts = np.bincount(window.ravel(), minlength=int(cfg["classes"]["num_classes"]))
    total = max(int(counts.sum()), 1)
    names = cfg["classes"]["names"]
    air_id = names.index("Air") if "Air" in names else None
    water_fraction = float(counts[air_id] / total) if air_id is not None else 0.0

    selected = int(np.argmax(counts))
    if air_id is not None and selected == air_id and water_fraction < 0.60:
        non_air_counts = counts.copy()
        non_air_counts[air_id] = -1
        selected = int(np.argmax(non_air_counts))

    info = _lulc_info(float(selected), cfg)
    info.update(
        {
            "center_class_id": center,
            "center_class_name": names[center] if 0 <= center < len(names) else None,
            "center_window_size": window_size,
            "center_window_water_fraction": round(water_fraction, 4),
            "center_window_class_counts": {names[i]: int(counts[i]) for i in range(min(len(names), len(counts)))},
        }
    )
    return info


def get_flood_score_live(lat: float, lon: float, cfg: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    refs = normalization_reference(str(config_path))
    raw = gee_raw_values(lat, lon, cfg)
    lulc = predict_lulc_on_the_fly(lat, lon, cfg, str(config_path))
    breakdown = {
        "elevation": _elevation_score(raw.get("elevation_m"), refs["elevation"]["max"]),
        "slope": _linear_score(raw.get("slope_deg"), refs["slope"]["p02"], refs["slope"]["p98"], inverse=True),
        "rainfall": _linear_score(raw.get("rainfall_mm_year"), refs["rainfall"]["p02"], refs["rainfall"]["p98"], inverse=False),
        "river_distance": _linear_score(raw.get("river_distance_m"), refs["river_distance"]["p02"], refs["river_distance"]["p98"], inverse=True),
        "lulc": lulc["score"],
        "population": _linear_score(raw.get("population_density"), refs["population"]["p02"], refs["population"]["p98"], inverse=False),
    }
    warnings = []
    if raw.get("river_distance_warning"):
        warnings.append(str(raw["river_distance_warning"]))
    return _format_result(lat, lon, cfg, breakdown, lulc, "gee_on_the_fly", raw_values=raw, warnings=warnings)


def get_flood_score(
    lat: float,
    lon: float,
    config_path: str | Path = "config/config.yaml",
    mode: str = "auto",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Return point-level FSI result.

    Modes:
    - ``live``: fetch physical layers and Sentinel-2 from GEE, run U-Net, compute FSI.
    - ``local``: sample Week-1 raster artifacts.
    - ``auto``: try ``live`` when credentials are available, otherwise use ``local``.
    """

    cfg = load_config(config_path)
    requested_mode = mode
    live_available = bool(os.getenv("GEE_SERVICE_ACCOUNT_KEY") or os.getenv("EARTHENGINE_PROJECT") or cfg["gee"].get("project_id"))
    if mode == "auto":
        mode = "live" if live_available else "local"

    key = _cache_key(lat, lon, mode, config_path)
    if use_cache:
        cached = _cache_get(key, cfg)
        if cached is not None:
            cached["cached"] = True
            return cached

    warnings: list[str] = []
    try:
        if mode == "live":
            result = get_flood_score_live(lat, lon, cfg, config_path)
        elif mode == "local":
            result = get_flood_score_local(lat, lon, cfg)
        else:
            raise ValueError("mode must be one of: auto, live, local")
    except Exception as exc:
        if requested_mode == "live":
            raise
        warnings.append(f"GEE on-the-fly failed, local raster fallback used: {exc}")
        result = get_flood_score_local(lat, lon, cfg)
        result["computation_mode"] = "local_raster_fallback"
        result["warnings"] = warnings + result.get("warnings", [])

    result["cache_key"] = key
    if use_cache:
        _cache_set(key, result, cfg)
    return result
