from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_json, resolve_path
from .dataset import FloodTileDataset
from .metrics import iou_from_confusion, update_confusion_matrix
from .model import build_unet


COLORS = np.array(
    [
        [180, 180, 180],
        [52, 145, 73],
        [214, 185, 84],
        [49, 120, 185],
        [196, 155, 95],
    ],
    dtype=np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(COLORS))
    out[valid] = COLORS[mask[valid]]
    return out


def preferred_list_path(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["_project_root"])
    clean_value = cfg["paths"].get("val_list_clean")
    if clean_value:
        clean_path = resolve_path(clean_value, root)
        if clean_path.exists():
            return clean_path
    return resolve_path(cfg["paths"]["val_list"], root)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    for normalize, filename, title in [
        (False, "confusion_matrix.png", "Confusion Matrix"),
        (True, "confusion_matrix_normalized.png", "Confusion Matrix Normalized by True Class"),
    ]:
        values = cm.astype(np.float64)
        fmt = "d"
        if normalize:
            row_sum = values.sum(axis=1, keepdims=True)
            values = np.divide(values, row_sum, out=np.zeros_like(values), where=row_sum > 0)
            fmt = ".2f"
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(values, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_yticklabels(class_names)
        threshold = values.max() * 0.55 if values.size else 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                text = format(int(cm[i, j]), "d") if not normalize else format(values[i, j], fmt)
                ax.text(j, i, text, ha="center", va="center", color="white" if values[i, j] > threshold else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)


def resolve_tile_image_path(path_value: str, processed_dir: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    name = PureWindowsPath(path_value).name if "\\" in path_value else path.name
    fallback = processed_dir / "images" / name
    if fallback.exists():
        return fallback
    return path


def write_prediction_mosaic(
    cfg: dict[str, Any],
    model: torch.nn.Module,
    stats: dict[str, Any],
    device: torch.device,
) -> Path:
    root = Path(cfg["_project_root"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"], root)
    tile_index = pd.read_csv(processed_dir / "tile_index.csv")
    out_path = resolve_path(cfg["evaluation"]["prediction_mosaic_tif"], root)
    ignore_index = int(cfg["preprocessing"]["ignore_index"])
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]

    with rasterio.open(resolve_path(cfg["paths"]["s2_tif"], root)) as src:
        profile = src.profile.copy()
    mosaic = np.full((profile["height"], profile["width"]), ignore_index, dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for row in tqdm(tile_index.itertuples(index=False), total=len(tile_index), desc="Writing prediction mosaic"):
            image_path = resolve_tile_image_path(row.image_path, processed_dir)
            image = np.clip(np.load(image_path).astype(np.float32), 0.0, 1.0)
            image = (image - mean) / std
            tensor = torch.from_numpy(image).unsqueeze(0).to(device)
            pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            r = int(row.row)
            c = int(row.col)
            h, w = pred.shape
            mosaic[r : r + h, c : c + w] = pred

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=ignore_index, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic, 1)
    return out_path


def evaluate_from_config(cfg: dict[str, Any], checkpoint: str | Path) -> Path:
    root = Path(cfg["_project_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_json(resolve_path(cfg["paths"]["dataset_stats"], root))
    val_ds = FloodTileDataset(preferred_list_path(cfg), stats, augment=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"])

    model = build_unet(cfg["training"]["encoder"], None, cfg["training"]["in_channels"], cfg["classes"]["num_classes"]).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = resolve_path(cfg["evaluation"]["output_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = np.zeros((cfg["classes"]["num_classes"], cfg["classes"]["num_classes"]), dtype=np.int64)
    saved = 0
    max_png = int(cfg["evaluation"]["representative_png_count"])

    with torch.no_grad():
        for image, mask in tqdm(val_loader, desc="Evaluating"):
            image = image.to(device)
            logits = model(image)
            pred = logits.argmax(dim=1).cpu()
            cm = update_confusion_matrix(cm, pred, mask, cfg["classes"]["num_classes"], cfg["preprocessing"]["ignore_index"])
            for i in range(pred.shape[0]):
                if saved >= max_png:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(colorize(mask[i].numpy()))
                axes[0].set_title("Ground truth")
                axes[0].axis("off")
                axes[1].imshow(colorize(pred[i].numpy()))
                axes[1].set_title("Prediction")
                axes[1].axis("off")
                fig.tight_layout()
                fig.savefig(out_dir / f"prediction_vs_gt_{saved+1:02d}.png", dpi=160)
                plt.close(fig)
                saved += 1

    pd.DataFrame(cm, index=cfg["classes"]["names"], columns=cfg["classes"]["names"]).to_csv(out_dir / "confusion_matrix.csv")
    plot_confusion_matrix(cm, cfg["classes"]["names"], out_dir)
    ious = iou_from_confusion(cm)
    pd.DataFrame({"class": cfg["classes"]["names"], "iou": ious}).to_csv(out_dir / "iou_per_class.csv", index=False)
    write_prediction_mosaic(cfg, model, stats, device)
    return out_dir
