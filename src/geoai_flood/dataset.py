from __future__ import annotations

import random
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def read_list(path: str | Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_path, mask_path = line.split(",", maxsplit=1)
            pairs.append((image_path, mask_path))
    return pairs


def jitter_rgb(image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    out = image.copy()
    rgb = out[:3]
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        rgb = rgb * factor
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = rgb.mean(axis=(1, 2), keepdims=True)
        rgb = (rgb - mean) * factor + mean
    out[:3] = np.clip(rgb, 0.0, 1.0)
    return out


class FloodTileDataset(Dataset):
    def __init__(
        self,
        list_path: str | Path,
        stats: dict[str, Any],
        augment: bool,
        augment_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.pairs = read_list(list_path)
        self.tile_root = Path(list_path).parent
        self.mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
        self.augment = augment
        self.augment_cfg = augment_cfg or {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _resolve_tile_path(self, path_value: str, subdir: str) -> Path:
        path = Path(path_value)
        if path.exists():
            return path
        name = PureWindowsPath(path_value).name if "\\" in path_value else path.name
        fallback = self.tile_root / subdir / name
        if fallback.exists():
            return fallback
        return path

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = np.load(self._resolve_tile_path(image_path, "images")).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        mask = np.load(self._resolve_tile_path(mask_path, "masks")).astype(np.int64)

        if self.augment:
            if self.augment_cfg.get("hflip", True) and random.random() < 0.5:
                image = image[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if self.augment_cfg.get("vflip", True) and random.random() < 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[::-1, :].copy()
            if self.augment_cfg.get("rot90", True):
                k = random.randint(0, 3)
                if k:
                    image = np.rot90(image, k=k, axes=(1, 2)).copy()
                    mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
            jitter = self.augment_cfg.get("color_jitter_rgb")
            if jitter:
                image = jitter_rgb(image, float(jitter.get("brightness", 0.0)), float(jitter.get("contrast", 0.0)))

        image = np.clip(image, 0.0, 1.0)
        image = (image - self.mean) / self.std
        return torch.from_numpy(image), torch.from_numpy(mask)
