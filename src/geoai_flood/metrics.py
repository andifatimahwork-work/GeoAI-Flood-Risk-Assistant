from __future__ import annotations

import numpy as np
import torch


def update_confusion_matrix(cm: np.ndarray, pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> np.ndarray:
    pred_np = pred.detach().cpu().numpy().ravel()
    target_np = target.detach().cpu().numpy().ravel()
    valid = target_np != ignore_index
    pred_np = pred_np[valid]
    target_np = target_np[valid]
    encoded = target_np * num_classes + pred_np
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    return cm + counts.reshape(num_classes, num_classes)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    return np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
