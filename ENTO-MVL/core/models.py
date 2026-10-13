from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _resnet50_weights(pretrained: bool):
    return tv_models.ResNet50_Weights.DEFAULT if pretrained else None


class CleanResNet50(nn.Module):
    """Standard ResNet50 baseline without ENTO/MVL modules."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        network = tv_models.resnet50(weights=_resnet50_weights(pretrained))
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

    def forward_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        value = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        f1 = self.layer1(value)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {"f2": f2, "f3": f3, "f4": f4}

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.forward_features(images)
        embedding = self.pool(features["f4"]).flatten(1)
        logits = self.classifier(embedding)
        return {
            "logits": logits,
            "logits_base": logits,
            "z_base": embedding,
            "z_final": embedding,
            **features,
        }


def extract_checkpoint_state(checkpoint, prefer_ema: bool = True) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    keys = (
        ["ema_model", "model", "raw_model", "state_dict"]
        if prefer_ema
        else ["model", "raw_model", "ema_model", "state_dict"]
    )
    for key in keys:
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    return checkpoint


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def load_checkpoint_state(path: str, prefer_ema: bool = True):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = strip_module_prefix(extract_checkpoint_state(checkpoint, prefer_ema=prefer_ema))
    return checkpoint, state


def load_clean_checkpoint(model: CleanResNet50, path: str, prefer_ema: bool = True):
    _, state = load_checkpoint_state(path, prefer_ema)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical = [
        key
        for key in missing
        if key.startswith(("conv1", "bn1", "layer", "classifier"))
    ]
    if critical or unexpected:
        raise RuntimeError(
            f"Clean checkpoint mismatch. Critical missing={critical[:20]}, "
            f"unexpected={unexpected[:20]}"
        )
    return missing, unexpected
