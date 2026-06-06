from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm

from .config import load_json, resolve_path, write_json


@dataclass
class TileRecord:
    image_path: str
    mask_path: str
    row: int
    col: int
    class_counts: list[int]


def read_raster(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read()
        profile = src.profile.copy()
    return arr, profile


def read_single_band_matched(path: Path, ref_profile: dict[str, Any], resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as src:
        dst = np.full((ref_profile["height"], ref_profile["width"]), 0, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=resampling,
        )
    return dst


def remap_esa(esa: np.ndarray, remap_cfg: dict[str, Any], ignore_index: int) -> np.ndarray:
    mask = np.full(esa.shape, ignore_index, dtype=np.uint8)
    for src_value, dst_value in remap_cfg["esa_to_custom"].items():
        mask[esa == int(src_value)] = int(dst_value)
    return mask


def apply_informal_override(mask: np.ndarray, informal: np.ndarray | None, remap_cfg: dict[str, Any]) -> np.ndarray:
    if informal is None:
        return mask
    override = remap_cfg["informal_override"]
    positives = set(int(v) for v in override["mask_positive_values"])
    positive_mask = np.isin(informal, list(positives))
    out = mask.copy()
    out[positive_mask] = int(override["target_class_id"])
    return out


def class_histogram(mask: np.ndarray, num_classes: int, ignore_index: int) -> list[int]:
    valid = mask != ignore_index
    if not valid.any():
        return [0] * num_classes
    return np.bincount(mask[valid].ravel(), minlength=num_classes)[:num_classes].astype(int).tolist()


def valid_fraction(image: np.ndarray, mask: np.ndarray, ignore_index: int) -> float:
    image_ok = np.isfinite(image).all(axis=0)
    mask_ok = mask != ignore_index
    return float((image_ok & mask_ok).mean())


def greedy_multilabel_split(records: list[TileRecord], train_fraction: float, num_classes: int, seed: int) -> tuple[list[TileRecord], list[TileRecord]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    total = np.asarray([r.class_counts for r in shuffled], dtype=np.float64).sum(axis=0)
    target_val = total * (1.0 - train_fraction)
    val_counts = np.zeros(num_classes, dtype=np.float64)
    val: list[TileRecord] = []
    train: list[TileRecord] = []
    max_val_tiles = round(len(shuffled) * (1.0 - train_fraction))

    def rarity_score(record: TileRecord) -> float:
        counts = np.asarray(record.class_counts, dtype=np.float64)
        rarity = np.where(total > 0, 1.0 / np.maximum(total, 1.0), 0.0)
        return float((counts * rarity).sum())

    for record in sorted(shuffled, key=rarity_score, reverse=True):
        counts = np.asarray(record.class_counts, dtype=np.float64)
        need = target_val - val_counts
        helps_val = np.dot(np.maximum(need, 0.0), counts) > 0
        if helps_val and len(val) < max_val_tiles:
            val.append(record)
            val_counts += counts
        else:
            train.append(record)
    return train, val


def write_list(path: Path, records: list[TileRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(f"{rec.image_path},{rec.mask_path}\n")


def compute_stats(records: list[TileRecord]) -> dict[str, list[float]]:
    sums = None
    sums_sq = None
    pixels = 0
    for rec in tqdm(records, desc="Computing train mean/std"):
        image = np.clip(np.load(rec.image_path).astype(np.float64), 0.0, 1.0)
        flat = image.reshape(image.shape[0], -1)
        if sums is None:
            sums = flat.sum(axis=1)
            sums_sq = (flat**2).sum(axis=1)
        else:
            sums += flat.sum(axis=1)
            sums_sq += (flat**2).sum(axis=1)
        pixels += flat.shape[1]
    mean = sums / pixels
    var = np.maximum(sums_sq / pixels - mean**2, 1e-12)
    return {"mean": mean.tolist(), "std": np.sqrt(var).tolist()}


def make_tiles(cfg: dict[str, Any]) -> tuple[list[TileRecord], list[TileRecord]]:
    root = Path(cfg["_project_root"])
    paths = cfg["paths"]
    prep = cfg["preprocessing"]
    classes = cfg["classes"]
    ignore_index = int(prep["ignore_index"])
    tile_size = int(prep["tile_size"])
    stride = int(prep["stride"])
    out_dir = resolve_path(paths["processed_dir"], root)
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    s2, s2_profile = read_raster(resolve_path(paths["s2_tif"], root))
    esa = read_single_band_matched(resolve_path(paths["esa_tif"], root), s2_profile, Resampling.nearest)
    remap_cfg = load_json(resolve_path(paths["remap_config"], root))
    mask = remap_esa(esa, remap_cfg, ignore_index)

    informal_path = paths.get("informal_mask")
    informal = None
    if informal_path:
        informal = read_single_band_matched(resolve_path(informal_path, root), s2_profile, Resampling.nearest)
    mask = apply_informal_override(mask, informal, remap_cfg)

    records: list[TileRecord] = []
    height, width = mask.shape
    for row in tqdm(range(0, height - tile_size + 1, stride), desc="Writing tiles"):
        for col in range(0, width - tile_size + 1, stride):
            image_tile = s2[:, row : row + tile_size, col : col + tile_size].astype(cfg["preprocessing"]["image_dtype"])
            mask_tile = mask[row : row + tile_size, col : col + tile_size]
            if valid_fraction(image_tile, mask_tile, ignore_index) < float(prep["min_valid_fraction"]):
                continue
            stem = f"tile_r{row:06d}_c{col:06d}"
            image_path = image_dir / f"{stem}.npy"
            mask_path = mask_dir / f"{stem}.npy"
            np.save(image_path, image_tile)
            np.save(mask_path, mask_tile)
            records.append(
                TileRecord(
                    image_path=str(image_path),
                    mask_path=str(mask_path),
                    row=row,
                    col=col,
                    class_counts=class_histogram(mask_tile, classes["num_classes"], ignore_index),
                )
            )

    if not records:
        raise RuntimeError("No valid tiles created. Check raster paths, extent overlap, and min_valid_fraction.")

    train, val = greedy_multilabel_split(
        records,
        train_fraction=float(prep["train_fraction"]),
        num_classes=int(classes["num_classes"]),
        seed=int(cfg["project"]["seed"]),
    )

    write_list(resolve_path(paths["train_list"], root), train)
    write_list(resolve_path(paths["val_list"], root), val)

    stats = compute_stats(train)
    write_json(resolve_path(paths["dataset_stats"], root), stats)

    shutil.copy2(resolve_path(paths["remap_config"], root), resolve_path(paths["output_dir"], root) / "remap_config.json")
    with (out_dir / "tile_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "row", "col", *classes["names"]])
        for rec in records:
            writer.writerow([rec.image_path, rec.mask_path, rec.row, rec.col, *rec.class_counts])
    return train, val
