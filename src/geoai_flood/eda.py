from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from .config import load_json, resolve_path
from .dataset import read_list


def load_tile_records(cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["_project_root"])
    tile_index_path = resolve_path(cfg["paths"]["processed_dir"], root) / "tile_index.csv"
    df = pd.read_csv(tile_index_path)
    split_lookup = {}
    for split_name, key in [("train", "train_list"), ("val", "val_list")]:
        for image_path, mask_path in read_list(resolve_path(cfg["paths"][key], root)):
            split_lookup[image_path] = split_name
            split_lookup[mask_path] = split_name
    df["split"] = df["image_path"].map(split_lookup).fillna("unknown")
    return df


def class_counts_from_masks(records: pd.DataFrame, num_classes: int, ignore_index: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for row in tqdm(records.itertuples(index=False), total=len(records), desc="Counting classes"):
        mask = np.load(row.mask_path)
        valid = mask != ignore_index
        counts += np.bincount(mask[valid].ravel(), minlength=num_classes)[:num_classes]
    return counts


def normalized_inverse_weights(counts: np.ndarray) -> np.ndarray:
    nonzero = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)
    weights[nonzero] = counts[nonzero].sum() / counts[nonzero]
    weights[nonzero] = weights[nonzero] / max(weights[nonzero].mean(), 1e-12)
    return weights


def write_class_weights(cfg: dict[str, Any], train_records: pd.DataFrame, out_dir: Path) -> Path:
    num_classes = int(cfg["classes"]["num_classes"])
    ignore_index = int(cfg["preprocessing"]["ignore_index"])
    counts = class_counts_from_masks(train_records, num_classes, ignore_index)
    weights = normalized_inverse_weights(counts)
    total = max(float(counts.sum()), 1.0)
    zero_pixel_class_ids = [int(i) for i, count in enumerate(counts) if count == 0]
    payload = {
        "class_names": cfg["classes"]["names"],
        "pixel_counts": {name: int(counts[i]) for i, name in enumerate(cfg["classes"]["names"])},
        "pixel_fraction": {name: float(counts[i] / total) for i, name in enumerate(cfg["classes"]["names"])},
        "weights": {name: float(weights[i]) for i, name in enumerate(cfg["classes"]["names"])},
        "weights_list": weights.tolist(),
        "active_class_ids": [int(i) for i, count in enumerate(counts) if count > 0],
        "zero_pixel_class_ids": zero_pixel_class_ids,
        "zero_pixel_class_names": [cfg["classes"]["names"][i] for i in zero_pixel_class_ids],
        "method": "inverse_frequency_normalized_to_mean_1",
        "computed_from": "exact pixel counts on clean train split after EDA filtering and stratified tile split",
    }
    out_path = out_dir / "class_weights.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def write_eda_warnings(cfg: dict[str, Any], class_weights_path: Path, out_dir: Path) -> Path:
    payload = json.loads(class_weights_path.read_text(encoding="utf-8"))
    warnings = []
    zero_names = payload.get("zero_pixel_class_names", [])
    if zero_names:
        warnings.append(
            "Zero-pixel classes detected and excluded from weighted loss/Dice averaging: "
            + ", ".join(zero_names)
        )
    warnings.append(
        "Sawah and Vegetasi may overlap spectrally because ESA WorldCover pseudo-labels are ambiguous in mixed vegetation/agriculture areas; lower Sawah IoU is a known limitation."
    )
    warnings.append(
        "Informal settlement is dropped from U-Net labels because ESA WorldCover has no explicit Informal class; dense-settlement risk is represented later with WorldPop in the FSI formula."
    )
    out_path = out_dir / "eda_warnings.txt"
    out_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
    return out_path


def plot_class_distribution(cfg: dict[str, Any], records: pd.DataFrame, out_dir: Path) -> Path:
    num_classes = int(cfg["classes"]["num_classes"])
    ignore_index = int(cfg["preprocessing"]["ignore_index"])
    rows = []
    for split in ["train", "val"]:
        subset = records[records["split"] == split]
        counts = class_counts_from_masks(subset, num_classes, ignore_index)
        total = max(float(counts.sum()), 1.0)
        for i, name in enumerate(cfg["classes"]["names"]):
            rows.append({"split": split, "class": name, "pixels": counts[i], "fraction": counts[i] / total})
    dist = pd.DataFrame(rows)
    dist.to_csv(out_dir / "class_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(num_classes)
    width = 0.38
    for offset, split in [(-width / 2, "train"), (width / 2, "val")]:
        values = dist[dist["split"] == split]["fraction"].to_numpy()
        ax.bar(x + offset, values, width=width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg["classes"]["names"], rotation=25, ha="right")
    ax.set_ylabel("Pixel fraction")
    ax.set_title("LULC Class Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / "class_distribution.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def inspect_tile_quality(cfg: dict[str, Any], records: pd.DataFrame, out_dir: Path) -> tuple[Path, Path, Path]:
    ignore_index = int(cfg["preprocessing"]["ignore_index"])
    quality_cfg = cfg.get("eda", {})
    max_bad_fraction = float(quality_cfg.get("max_bad_pixel_fraction", 0.30))
    drop_single_class = bool(quality_cfg.get("drop_single_class_tiles", True))
    max_single_class_fraction = float(quality_cfg.get("max_single_class_fraction", 0.95))
    summary_rows = []
    bad_paths = []

    for row in tqdm(records.itertuples(index=False), total=len(records), desc="Checking tile quality"):
        image = np.load(row.image_path)
        mask = np.load(row.mask_path)
        bad_image_fraction = float((~np.isfinite(image).all(axis=0)).mean())
        nodata_fraction = float((mask == ignore_index).mean())
        valid_labels = mask[mask != ignore_index]
        unique_classes = np.unique(valid_labels)
        if len(valid_labels) == 0:
            dominant_class_fraction = 0.0
            dominant_class = -1
        else:
            counts = np.bincount(valid_labels.ravel(), minlength=int(cfg["classes"]["num_classes"]))[: int(cfg["classes"]["num_classes"])]
            dominant_class = int(np.argmax(counts))
            dominant_class_fraction = float(counts[dominant_class] / max(counts.sum(), 1))
        mostly_single_class = dominant_class_fraction > max_single_class_fraction
        is_bad = (
            bad_image_fraction > max_bad_fraction
            or nodata_fraction > max_bad_fraction
            or (drop_single_class and mostly_single_class)
        )
        if is_bad:
            bad_paths.append((row.image_path, row.mask_path))
        summary_rows.append(
            {
                "image_path": row.image_path,
                "mask_path": row.mask_path,
                "split": row.split,
                "bad_image_fraction": bad_image_fraction,
                "nodata_fraction": nodata_fraction,
                "unique_class_count": int(len(unique_classes)),
                "dominant_class": dominant_class,
                "dominant_class_name": cfg["classes"]["names"][dominant_class] if dominant_class >= 0 else "none",
                "dominant_class_fraction": dominant_class_fraction,
                "mostly_single_class": bool(mostly_single_class),
                "is_bad": bool(is_bad),
            }
        )

    quality = pd.DataFrame(summary_rows)
    quality_path = out_dir / "tile_quality.csv"
    quality.to_csv(quality_path, index=False)

    bad_path = out_dir / "bad_tiles.txt"
    with bad_path.open("w", encoding="utf-8") as f:
        for image_path, mask_path in bad_paths:
            f.write(f"{image_path},{mask_path}\n")

    return bad_path, resolve_path(cfg["paths"]["train_list_clean"], Path(cfg["_project_root"])), resolve_path(cfg["paths"]["val_list_clean"], Path(cfg["_project_root"]))


def dominant_class_stratified_split(cfg: dict[str, Any], records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fraction = float(cfg["preprocessing"]["train_fraction"])
    seed = int(cfg["project"]["seed"])
    class_cols = cfg["classes"]["names"]
    split_frames = []
    for class_id, class_name in enumerate(class_cols):
        class_records = records[records["dominant_class"] == class_id]
        if class_records.empty:
            continue
        val_count = round(len(class_records) * (1.0 - train_fraction))
        if len(class_records) > 1:
            val_count = min(max(val_count, 1), len(class_records) - 1)
        else:
            val_count = 0
        shuffled = class_records.sample(frac=1.0, random_state=seed + class_id)
        val_part = shuffled.iloc[:val_count].copy()
        train_part = shuffled.iloc[val_count:].copy()
        val_part["split"] = "val"
        train_part["split"] = "train"
        split_frames.extend([train_part, val_part])
    split_records = pd.concat(split_frames, ignore_index=True) if split_frames else records.copy()
    train_records = split_records[split_records["split"] == "train"].copy()
    val_records = split_records[split_records["split"] == "val"].copy()
    return train_records, val_records


def write_clean_lists(cfg: dict[str, Any], train_records: pd.DataFrame, val_records: pd.DataFrame) -> tuple[Path, Path]:
    root = Path(cfg["_project_root"])
    clean_train_path = resolve_path(cfg["paths"]["train_list_clean"], root)
    clean_val_path = resolve_path(cfg["paths"]["val_list_clean"], root)
    for path, records in [(clean_train_path, train_records), (clean_val_path, val_records)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in records.itertuples(index=False):
                f.write(f"{row.image_path},{row.mask_path}\n")
    return clean_train_path, clean_val_path


def write_clipped_dataset_stats(cfg: dict[str, Any], train_records: pd.DataFrame, out_dir: Path) -> Path:
    sums = None
    sums_sq = None
    pixels = 0
    for row in tqdm(train_records.itertuples(index=False), total=len(train_records), desc="Computing clipped train mean/std"):
        image = np.clip(np.load(row.image_path).astype(np.float64), 0.0, 1.0)
        flat = image.reshape(image.shape[0], -1)
        if sums is None:
            sums = flat.sum(axis=1)
            sums_sq = (flat**2).sum(axis=1)
        else:
            sums += flat.sum(axis=1)
            sums_sq += (flat**2).sum(axis=1)
        pixels += flat.shape[1]
    mean = sums / max(pixels, 1)
    var = np.maximum(sums_sq / max(pixels, 1) - mean**2, 1e-12)
    payload = {"mean": mean.tolist(), "std": np.sqrt(var).tolist(), "note": "Computed on clean train split after clipping bands to [0, 1]."}
    root = Path(cfg["_project_root"])
    dataset_stats_path = resolve_path(cfg["paths"]["dataset_stats"], root)
    dataset_stats_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    eda_copy = out_dir / "dataset_stats_clipped.json"
    eda_copy.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dataset_stats_path


def plot_band_stats(cfg: dict[str, Any], records: pd.DataFrame, out_dir: Path) -> Path:
    rng = random.Random(int(cfg["project"]["seed"]))
    sample_count = min(int(cfg.get("eda", {}).get("band_histogram_tiles", 50)), len(records))
    sampled = records.sample(n=sample_count, random_state=rng.randint(0, 10_000)) if sample_count else records
    band_values = [[] for _ in cfg["gee"]["bands"]]
    for row in tqdm(sampled.itertuples(index=False), total=len(sampled), desc="Sampling band histograms"):
        image = np.clip(np.load(row.image_path), 0.0, 1.0)
        for band_i in range(image.shape[0]):
            values = image[band_i].ravel()
            values = values[np.isfinite(values)]
            if len(values) > 12000:
                values = rng.sample(values.tolist(), 12000)
            band_values[band_i].extend(values)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for i, band in enumerate(cfg["gee"]["bands"]):
        axes[i].hist(band_values[i], bins=60, color="#3A7CA5", alpha=0.85)
        axes[i].set_title(f"{band} histogram")
        axes[i].set_xlabel("Reflectance")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(alpha=0.2)
    fig.tight_layout()
    out_path = out_dir / "band_stats_plot.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def robust_rgb(image: np.ndarray) -> np.ndarray:
    rgb = np.stack([image[2], image[1], image[0]], axis=-1)
    lo, hi = np.nanpercentile(rgb, [2, 98])
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def plot_sample_visuals(cfg: dict[str, Any], records: pd.DataFrame, out_dir: Path) -> Path:
    root = Path(cfg["_project_root"])
    count = min(int(cfg.get("eda", {}).get("sample_visual_tiles", 12)), len(records))
    sampled = records.sample(n=count, random_state=int(cfg["project"]["seed"])) if count else records
    colors = np.array(
        [
            [180, 180, 180],
            [52, 145, 73],
            [214, 185, 84],
            [49, 120, 185],
            [196, 155, 95],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(resolve_path(cfg["paths"]["esa_tif"], root)) as src:
        esa = src.read(1)

    fig, axes = plt.subplots(count, 4, figsize=(12, max(3, count * 2.3)))
    if count == 1:
        axes = np.expand_dims(axes, 0)
    for ax_row, row in zip(axes, sampled.itertuples(index=False)):
        image = np.clip(np.load(row.image_path), 0.0, 1.0)
        mask = np.load(row.mask_path)
        r, c = int(row.row), int(row.col)
        esa_tile = esa[r : r + mask.shape[0], c : c + mask.shape[1]]
        ax_row[0].imshow(robust_rgb(image))
        ax_row[0].set_title("RGB")
        ax_row[1].imshow(image[3], cmap="gray")
        ax_row[1].set_title("NIR B8")
        ax_row[2].imshow(esa_tile, cmap="tab20")
        ax_row[2].set_title("ESA label")
        remapped = mask.copy()
        remapped[remapped >= len(colors)] = len(colors) - 1
        ax_row[3].imshow(colors[remapped])
        ax_row[3].set_title("Remapped")
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    out_path = out_dir / "sample_visual_inspection.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_red_nir_scatter(cfg: dict[str, Any], records: pd.DataFrame, out_dir: Path) -> Path:
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    max_pixels = int(cfg.get("eda", {}).get("scatter_max_pixels", 60000))
    per_tile = max(max_pixels // max(len(records), 1), 100)
    red_values = []
    nir_values = []
    labels = []
    for row in tqdm(records.itertuples(index=False), total=len(records), desc="Sampling B4/B8 scatter"):
        image = np.clip(np.load(row.image_path), 0.0, 1.0)
        mask = np.load(row.mask_path)
        valid = mask != int(cfg["preprocessing"]["ignore_index"])
        idx = np.flatnonzero(valid.ravel())
        if len(idx) == 0:
            continue
        take = min(per_tile, len(idx))
        sampled_idx = rng.choice(idx, size=take, replace=False)
        red_values.append(image[2].ravel()[sampled_idx])
        nir_values.append(image[3].ravel()[sampled_idx])
        labels.append(mask.ravel()[sampled_idx])
    red = np.concatenate(red_values)
    nir = np.concatenate(nir_values)
    label = np.concatenate(labels)
    if len(red) > max_pixels:
        idx = rng.choice(np.arange(len(red)), size=max_pixels, replace=False)
        red, nir, label = red[idx], nir[idx], label[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    for class_id, name in enumerate(cfg["classes"]["names"]):
        m = label == class_id
        if m.any():
            ax.scatter(red[m], nir[m], s=2, alpha=0.25, label=name)
    ax.set_xlabel("Red B4 reflectance")
    ax.set_ylabel("NIR B8 reflectance")
    ax.set_title("B8 vs B4 Feature Space")
    ax.legend(markerscale=4, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = out_dir / "red_nir_scatter.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def run_eda(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["_project_root"])
    out_dir = resolve_path(cfg["paths"]["eda_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_tile_records(cfg)
    bad_tiles, clean_train, clean_val = inspect_tile_quality(cfg, records, out_dir)
    bad_images = {
        line.split(",", maxsplit=1)[0]
        for line in bad_tiles.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    clean_records = records[~records["image_path"].isin(bad_images)].copy()
    class_cols = cfg["classes"]["names"]
    clean_records["dominant_class"] = clean_records[class_cols].to_numpy().argmax(axis=1)
    clean_records["dominant_class_name"] = clean_records["dominant_class"].map(lambda i: cfg["classes"]["names"][int(i)])
    clean_records["dominant_class_fraction"] = clean_records[class_cols].max(axis=1) / clean_records[class_cols].sum(axis=1).clip(lower=1)
    train_records, val_records = dominant_class_stratified_split(cfg, clean_records)
    clean_train, clean_val = write_clean_lists(cfg, train_records, val_records)
    split_summary = (
        pd.concat([train_records, val_records], ignore_index=True)
        .groupby(["split", "dominant_class_name"])
        .size()
        .reset_index(name="tiles")
    )
    split_summary.to_csv(out_dir / "dominant_class_split_summary.csv", index=False)

    outputs = {
        "class_distribution": str(plot_class_distribution(cfg, pd.concat([train_records, val_records], ignore_index=True), out_dir)),
        "class_weights": str(write_class_weights(cfg, train_records, out_dir)),
    }
    warnings_path = write_eda_warnings(cfg, Path(outputs["class_weights"]), out_dir)
    outputs.update(
        {
            "bad_tiles": str(bad_tiles),
            "clean_train_list": str(clean_train),
            "clean_val_list": str(clean_val),
            "dataset_stats": str(write_clipped_dataset_stats(cfg, train_records, out_dir)),
            "split_summary": str(out_dir / "dominant_class_split_summary.csv"),
            "warnings": str(warnings_path),
            "band_stats_plot": str(plot_band_stats(cfg, train_records, out_dir)),
            "sample_visual_inspection": str(plot_sample_visuals(cfg, pd.concat([train_records, val_records], ignore_index=True), out_dir)),
            "red_nir_scatter": str(plot_red_nir_scatter(cfg, train_records, out_dir)),
        }
    )
    with (out_dir / "eda_outputs.json").open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    return out_dir
