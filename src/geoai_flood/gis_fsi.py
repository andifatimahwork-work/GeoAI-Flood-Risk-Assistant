from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .config import load_json, resolve_path


def read_reference(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        profile = src.profile.copy()
    return arr, profile


def read_matched(path: Path, ref_profile: dict[str, Any], resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    with rasterio.open(path) as src:
        dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype=np.float32)
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
        if src_nodata is not None:
            dst[dst == src_nodata] = np.nan
    return dst


def normalize(arr: np.ndarray, inverse: bool = False, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    valid = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    if not valid.any():
        return out
    lo, hi = np.nanpercentile(arr[valid], [lower, upper])
    scaled = (arr - lo) / max(hi - lo, 1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)
    if inverse:
        scaled = 1.0 - scaled
    out[valid] = scaled[valid]
    return out.astype(np.float32)


def normalize_elevation(elevation: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    method = cfg["fsi"].get("elevation_transform", "log_inverse")
    if method != "log_inverse":
        return normalize(elevation, inverse=True)

    valid = np.isfinite(elevation)
    out = np.full(elevation.shape, np.nan, dtype=np.float32)
    if not valid.any():
        return out

    elev = np.maximum(elevation.astype(np.float32), 0.0)
    max_elev = float(np.nanmax(elev[valid]))
    if max_elev <= 0:
        out[valid] = 1.0
        return out

    score = 1.0 - (np.log1p(elev) / np.log1p(max_elev))
    out[valid] = np.clip(score[valid], 0.0, 1.0)
    return out.astype(np.float32)


def derive_slope_score(elevation: np.ndarray, transform: Any) -> np.ndarray:
    xres = abs(transform.a)
    yres = abs(transform.e)
    gy, gx = np.gradient(elevation.astype(np.float32), yres, xres)
    slope = np.sqrt(gx**2 + gy**2)
    return normalize(slope, inverse=True)


def lulc_to_risk(lulc: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    names = cfg["classes"]["names"]
    scores_by_name = cfg["fsi"]["lulc_risk_scores"]
    out = np.full(lulc.shape, np.nan, dtype=np.float32)
    for class_id, name in enumerate(names):
        out[lulc == class_id] = float(scores_by_name[name])
    return out


def lulc_valid_mask(lulc: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    valid = np.isfinite(lulc)
    class_to_id = {name: i for i, name in enumerate(cfg["classes"]["names"])}
    for class_name in cfg["fsi"].get("mask_lulc_classes", []):
        class_id = class_to_id.get(class_name)
        if class_id is not None:
            valid &= lulc != class_id
    return valid


def esa_to_custom_lulc(esa: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    root = Path(cfg["_project_root"])
    remap_cfg = load_json(resolve_path(cfg["paths"]["remap_config"], root))
    ignore_index = int(cfg["preprocessing"]["ignore_index"])
    out = np.full(esa.shape, ignore_index, dtype=np.int16)
    for src_value, dst_value in remap_cfg["esa_to_custom"].items():
        out[esa == int(src_value)] = int(dst_value)
    return out


def write_like(path: Path, data: np.ndarray, ref_profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(data), data, -9999.0).astype(np.float32), 1)


def write_component_layers(component_dir: Path, layers: dict[str, np.ndarray], ref_profile: dict[str, Any]) -> None:
    component_dir.mkdir(parents=True, exist_ok=True)
    for name, data in layers.items():
        write_like(component_dir / f"{name}_score.tif", data, ref_profile)


def weighted_sum(layers: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    valid_mask = combined_valid_mask(layers)
    out = np.full(next(iter(layers.values())).shape, np.nan, dtype=np.float32)
    work = np.zeros(next(iter(layers.values())).shape, dtype=np.float32)
    total = sum(weights.values())
    for key, layer in layers.items():
        work += np.where(np.isfinite(layer), layer, 0.0).astype(np.float32) * float(weights[key])
    out[valid_mask] = work[valid_mask] / max(total, 1e-6)
    return out


def combined_valid_mask(layers: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.ones(next(iter(layers.values())).shape, dtype=bool)
    for layer in layers.values():
        valid &= np.isfinite(layer)
    return valid


def monte_carlo_ci_chunked(
    layers: dict[str, np.ndarray],
    weights: dict[str, float],
    iterations: int,
    perturbation_fraction: float,
    seed: int,
    row_chunk_size: int,
    extra_valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    keys = list(weights.keys())
    base = np.asarray([weights[k] for k in keys], dtype=np.float32)
    rng = np.random.default_rng(seed)
    sampled_weights = np.empty((iterations, len(keys)), dtype=np.float32)
    for i in range(iterations):
        noise = rng.uniform(1.0 - perturbation_fraction, 1.0 + perturbation_fraction, size=len(keys))
        sampled = base * noise
        sampled_weights[i] = sampled / sampled.sum()

    height, width = next(iter(layers.values())).shape
    valid_mask = combined_valid_mask(layers)
    if extra_valid_mask is not None:
        valid_mask &= extra_valid_mask
    ci_low = np.empty((height, width), dtype=np.float32)
    ci_high = np.empty((height, width), dtype=np.float32)
    row_chunk_size = max(1, int(row_chunk_size))

    for row_start in range(0, height, row_chunk_size):
        row_end = min(row_start + row_chunk_size, height)
        block_shape = (iterations, row_end - row_start, width)
        block_stack = np.empty(block_shape, dtype=np.float32)
        block_valid = valid_mask[row_start:row_end]
        for i in range(iterations):
            block = np.zeros((row_end - row_start, width), dtype=np.float32)
            for k, key in enumerate(keys):
                layer_block = layers[key][row_start:row_end]
                block += np.where(np.isfinite(layer_block), layer_block, 0.0).astype(np.float32) * sampled_weights[i, k]
            block[~block_valid] = np.nan
            block_stack[i] = block
        ci_low[row_start:row_end] = np.nanpercentile(block_stack, 5, axis=0).astype(np.float32)
        ci_high[row_start:row_end] = np.nanpercentile(block_stack, 95, axis=0).astype(np.float32)
    return ci_low, ci_high


def build_base_layers(cfg: dict[str, Any], ref_profile: dict[str, Any]) -> dict[str, np.ndarray]:
    root = Path(cfg["_project_root"])
    fsi_cfg = cfg["fsi"]
    elevation = read_matched(resolve_path(fsi_cfg["elevation_tif"], root), ref_profile)
    rainfall = read_matched(resolve_path(fsi_cfg["rainfall_tif"], root), ref_profile)
    river_distance = read_matched(resolve_path(fsi_cfg["river_distance_tif"], root), ref_profile)
    population = read_matched(resolve_path(fsi_cfg["population_tif"], root), ref_profile)
    if fsi_cfg.get("slope_tif") and resolve_path(fsi_cfg["slope_tif"], root).exists():
        slope_score = normalize(read_matched(resolve_path(fsi_cfg["slope_tif"], root), ref_profile), inverse=True)
    else:
        slope_score = derive_slope_score(elevation, ref_profile["transform"])
    return {
        "elevation": normalize_elevation(elevation, cfg),
        "slope": slope_score,
        "rainfall": normalize(rainfall, inverse=False),
        "river_distance": normalize(river_distance, inverse=True),
        "population": normalize(population, inverse=False),
    }


def fsi_for_lulc(
    cfg: dict[str, Any],
    ref_profile: dict[str, Any],
    base_layers: dict[str, np.ndarray],
    lulc_path: Path,
    output_path: Path,
    is_esa: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    lulc_float = read_matched(lulc_path, ref_profile, Resampling.nearest)
    lulc = np.where(np.isfinite(lulc_float), lulc_float, -9999).astype(np.int16)
    if is_esa:
        lulc = esa_to_custom_lulc(lulc, cfg)
    valid_lulc = lulc_valid_mask(lulc.astype(np.float32), cfg)
    layers = dict(base_layers)
    layers["lulc"] = lulc_to_risk(lulc, cfg)
    layers["lulc"][~valid_lulc] = np.nan
    fsi_cfg = cfg["fsi"]
    weights = {k: float(v) for k, v in fsi_cfg["weights"].items()}
    fsi = weighted_sum(layers, weights)
    return fsi, layers, valid_lulc


def build_fsi_from_config(cfg: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    root = Path(cfg["_project_root"])
    fsi_cfg = cfg["fsi"]
    _, ref_profile = read_reference(resolve_path(fsi_cfg["reference_tif"], root))
    base_layers = build_base_layers(cfg, ref_profile)
    write_component_layers(resolve_path(fsi_cfg["output_components_dir"], root), base_layers, ref_profile)
    fsi_path = resolve_path(fsi_cfg["output_fsi_tif"], root)
    baseline_path = resolve_path(fsi_cfg["output_baseline_fsi_tif"], root)
    fsi, proposed_layers, proposed_valid_lulc = fsi_for_lulc(
        cfg,
        ref_profile,
        base_layers,
        resolve_path(fsi_cfg["lulc_unet_tif"], root),
        fsi_path,
        is_esa=False,
    )
    baseline_fsi, baseline_layers, baseline_valid_lulc = fsi_for_lulc(
        cfg,
        ref_profile,
        base_layers,
        resolve_path(fsi_cfg["lulc_esa_tif"], root),
        baseline_path,
        is_esa=True,
    )
    common_valid_mask = np.isfinite(fsi) & np.isfinite(baseline_fsi) & proposed_valid_lulc & baseline_valid_lulc
    fsi = np.where(common_valid_mask, fsi, np.nan).astype(np.float32)
    baseline_fsi = np.where(common_valid_mask, baseline_fsi, np.nan).astype(np.float32)
    for key in proposed_layers:
        proposed_layers[key] = np.where(common_valid_mask, proposed_layers[key], np.nan).astype(np.float32)
    write_like(fsi_path, fsi, ref_profile)
    write_like(baseline_path, baseline_fsi, ref_profile)
    weights = {k: float(v) for k, v in fsi_cfg["weights"].items()}
    ci_low, ci_high = monte_carlo_ci_chunked(
        proposed_layers,
        weights,
        iterations=int(fsi_cfg["monte_carlo_iterations"]),
        perturbation_fraction=float(fsi_cfg["perturbation_fraction"]),
        seed=int(cfg["project"]["seed"]),
        row_chunk_size=int(fsi_cfg.get("monte_carlo_row_chunk_size", 16)),
        extra_valid_mask=common_valid_mask,
    )

    low_path = resolve_path(fsi_cfg["output_ci_low_tif"], root)
    high_path = resolve_path(fsi_cfg["output_ci_high_tif"], root)
    write_like(low_path, ci_low, ref_profile)
    write_like(high_path, ci_high, ref_profile)
    return fsi_path, baseline_path, low_path, high_path
