from __future__ import annotations

import segmentation_models_pytorch as smp
import torch


def build_unet(
    encoder: str,
    encoder_weights: str | None,
    in_channels: int,
    num_classes: int,
) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for param in encoder.parameters():
        param.requires_grad = trainable
