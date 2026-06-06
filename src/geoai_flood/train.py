from __future__ import annotations

import csv
import warnings
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .config import load_json, resolve_path
from .dataset import FloodTileDataset, read_list
from .metrics import iou_from_confusion, update_confusion_matrix
from .model import build_unet, set_encoder_trainable


def class_weights_from_masks(list_path: Path, num_classes: int, ignore_index: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, mask_path in read_list(list_path):
        mask = np.load(mask_path)
        valid = mask != ignore_index
        counts += np.bincount(mask[valid].ravel(), minlength=num_classes)[:num_classes]
    zero_classes = np.flatnonzero(counts == 0).tolist()
    if zero_classes:
        warnings.warn(f"Zero-pixel class ids in training set: {zero_classes}. They will be excluded from class weighting.")
    nonzero = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[nonzero] = counts[nonzero].sum() / counts[nonzero]
    weights[nonzero] = weights[nonzero] / max(weights[nonzero].mean(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def class_weight_payload(cfg: dict[str, Any], train_list: Path) -> dict[str, Any]:
    root = Path(cfg["_project_root"])
    class_weights_path = resolve_path(cfg["paths"].get("class_weights", ""), root)
    if class_weights_path.exists():
        return load_json(class_weights_path)
    weights = class_weights_from_masks(train_list, cfg["classes"]["num_classes"], cfg["preprocessing"]["ignore_index"])
    active_class_ids = [i for i, value in enumerate(weights.tolist()) if value > 0]
    return {"weights_list": weights.tolist(), "active_class_ids": active_class_ids}


def class_weights_from_payload(cfg: dict[str, Any], payload: dict[str, Any]) -> torch.Tensor:
    if "weights_list" in payload:
        return torch.tensor(payload["weights_list"], dtype=torch.float32)
    return torch.tensor([payload["weights"][name] for name in cfg["classes"]["names"]], dtype=torch.float32)


def active_class_ids_from_payload(cfg: dict[str, Any], payload: dict[str, Any]) -> list[int]:
    zero_names = payload.get("zero_pixel_class_names", [])
    if zero_names:
        warnings.warn("Zero-pixel classes excluded from Dice averaging: " + ", ".join(zero_names))
    if cfg["training"]["loss"].get("exclude_zero_pixel_classes", True):
        return [int(i) for i in payload.get("active_class_ids", list(range(cfg["classes"]["num_classes"])))]
    return list(range(cfg["classes"]["num_classes"]))


def preferred_list_path(cfg: dict[str, Any], clean_key: str, fallback_key: str) -> Path:
    root = Path(cfg["_project_root"])
    clean_value = cfg["paths"].get(clean_key)
    if clean_value:
        clean_path = resolve_path(clean_value, root)
        if clean_path.exists():
            return clean_path
    return resolve_path(cfg["paths"][fallback_key], root)


def resolve_mask_path(path_value: str, list_path: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    name = PureWindowsPath(path_value).name if "\\" in path_value else path.name
    fallback = list_path.parent / "masks" / name
    if fallback.exists():
        return fallback
    return path


def build_weighted_sampler(cfg: dict[str, Any], train_list: Path) -> WeightedRandomSampler | None:
    sampler_cfg = cfg["training"].get("sampler", {})
    if not sampler_cfg.get("enabled", False):
        return None

    class_to_id = {name: i for i, name in enumerate(cfg["classes"]["names"])}
    target_ids = [class_to_id[name] for name in sampler_cfg.get("target_classes", []) if name in class_to_id]
    if not target_ids:
        warnings.warn("Weighted sampler enabled but no valid target_classes were found in config.")
        return None

    min_fraction = float(sampler_cfg.get("min_target_fraction", 0.01))
    multiplier = float(sampler_cfg.get("target_multiplier", 3.0))
    ignore_index = int(cfg["preprocessing"]["ignore_index"])
    weights: list[float] = []
    boosted = 0

    for _, mask_path_value in read_list(train_list):
        mask = np.load(resolve_mask_path(mask_path_value, train_list))
        valid = mask != ignore_index
        valid_count = int(valid.sum())
        if valid_count == 0:
            weights.append(1.0)
            continue
        target_fraction = float(np.isin(mask[valid], target_ids).sum() / valid_count)
        if target_fraction >= min_fraction:
            weights.append(multiplier)
            boosted += 1
        else:
            weights.append(1.0)

    if boosted == 0:
        warnings.warn("Weighted sampler found no tiles containing target classes above min_target_fraction.")
        return None
    warnings.warn(f"Weighted sampler boosted {boosted}/{len(weights)} tiles for {sampler_cfg.get('target_classes')}.")
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    active_class_ids: list[int] | None = None,
) -> torch.Tensor:
    valid = target != ignore_index
    target_safe = target.clone()
    target_safe[~valid] = 0
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probs = probs * valid
    one_hot = one_hot * valid
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)
    dice = (2 * intersection + 1.0) / (union + 1.0)
    if active_class_ids:
        dice = dice[torch.tensor(active_class_ids, device=dice.device, dtype=torch.long)]
    return 1.0 - dice.mean()


def classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ce_weights: torch.Tensor,
    ignore_index: int,
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=ce_weights, ignore_index=ignore_index, reduction="none")
    valid = target != ignore_index
    if not valid.any():
        return ce.mean() * 0.0
    ce_valid = ce[valid]
    if loss_cfg.get("ce_type", "weighted_ce") == "focal":
        gamma = float(loss_cfg.get("focal_gamma", 2.0))
        pt = torch.exp(-ce_valid)
        return (((1.0 - pt) ** gamma) * ce_valid).mean()
    return ce_valid.mean()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict[str, Any],
    ce_weights: torch.Tensor,
    active_class_ids: list[int],
) -> tuple[float, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    num_classes = int(cfg["classes"]["num_classes"])
    ignore_index = int(cfg["preprocessing"]["ignore_index"])
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    loss_sum = 0.0
    n = 0
    loss_cfg = cfg["training"]["loss"]
    with torch.set_grad_enabled(is_train):
        for image, mask in tqdm(loader, leave=False):
            image = image.to(device)
            mask = mask.to(device)
            logits = model(image)
            ce = classification_loss(logits, mask, ce_weights, ignore_index, loss_cfg)
            dl = dice_loss(logits, mask, num_classes=num_classes, ignore_index=ignore_index, active_class_ids=active_class_ids)
            loss = float(loss_cfg["ce_weight"]) * ce + float(loss_cfg["dice_weight"]) * dl
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            pred = logits.argmax(dim=1)
            cm = update_confusion_matrix(cm, pred, mask, num_classes, ignore_index)
            loss_sum += float(loss.detach().cpu()) * image.size(0)
            n += image.size(0)
    return loss_sum / max(n, 1), cm


def save_checkpoint(path: Path, model: torch.nn.Module, epoch: int, phase: str, best_iou: float, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "phase": phase,
            "best_val_miou": best_iou,
            "class_names": cfg["classes"]["names"],
            "config": cfg,
        },
        path,
    )


def train_from_config(cfg: dict[str, Any], resume: str | None = None) -> Path:
    root = Path(cfg["_project_root"])
    train_cfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_json(resolve_path(cfg["paths"]["dataset_stats"], root))
    train_list = preferred_list_path(cfg, "train_list_clean", "train_list")
    val_list = preferred_list_path(cfg, "val_list_clean", "val_list")

    train_ds = FloodTileDataset(train_list, stats, augment=True, augment_cfg=train_cfg["augment"])
    val_ds = FloodTileDataset(val_list, stats, augment=False)
    train_sampler = build_weighted_sampler(cfg, train_list)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"], pin_memory=True)

    model = build_unet(train_cfg["encoder"], train_cfg["encoder_weights"], train_cfg["in_channels"], cfg["classes"]["num_classes"]).to(device)
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    weights_payload = class_weight_payload(cfg, train_list)
    ce_weights = class_weights_from_payload(cfg, weights_payload).to(device)
    active_class_ids = active_class_ids_from_payload(cfg, weights_payload)
    checkpoint_dir = resolve_path(train_cfg["checkpoint_dir"], root)
    best_path = checkpoint_dir / train_cfg["best_model_name"]
    log_path = resolve_path(train_cfg["log_csv"], root)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "phase", "train_loss", "val_loss", "val_miou", *[f"iou_{name}" for name in cfg["classes"]["names"]]]
    best_miou = -1.0
    patience_left = int(train_cfg["early_stopping_patience"])
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    global_epoch = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        phases = [
            ("warmup", int(train_cfg["warmup_epochs"]), float(train_cfg["warmup_lr"]), False),
            ("finetune", int(train_cfg["finetune_epochs"]), float(train_cfg["finetune_lr"]), True),
        ]
        for phase, epochs, lr, encoder_trainable in phases:
            set_encoder_trainable(model, encoder_trainable)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=float(train_cfg["weight_decay"]))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1)) if phase == "finetune" else None

            for _ in range(epochs):
                global_epoch += 1
                train_loss, _ = run_epoch(model, train_loader, optimizer, device, cfg, ce_weights, active_class_ids)
                val_loss, val_cm = run_epoch(model, val_loader, None, device, cfg, ce_weights, active_class_ids)
                if scheduler:
                    scheduler.step()
                ious = iou_from_confusion(val_cm)
                miou = float(np.nanmean(ious[active_class_ids])) if active_class_ids else float(np.nanmean(ious))
                row = {"epoch": global_epoch, "phase": phase, "train_loss": train_loss, "val_loss": val_loss, "val_miou": miou}
                row.update({f"iou_{name}": float(ious[i]) for i, name in enumerate(cfg["classes"]["names"])})
                writer.writerow(row)
                f.flush()

                if miou > best_miou + min_delta:
                    best_miou = miou
                    patience_left = int(train_cfg["early_stopping_patience"])
                    save_checkpoint(best_path, model, global_epoch, phase, best_miou, cfg)
                else:
                    patience_left -= 1

                if global_epoch % 10 == 0:
                    save_checkpoint(checkpoint_dir / f"epoch_{global_epoch:03d}.pth", model, global_epoch, phase, best_miou, cfg)
                if patience_left <= 0:
                    return best_path
    return best_path
