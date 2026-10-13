from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from noattr_modules import AutomaticRegionEncoder


class IndependentAutoRegionBranchResNet50(nn.Module):
    ENTO_SOURCE_LAYER = "layer2"
    ENTO_SOURCE_FEATURE = "f2"
    ENTO_SOURCE_CHANNELS = 512
    MVL_SOURCE_LAYER = "layer3"

    """A genuine no-attribute ENTO-only or MVL-only ResNet50 plugin.

    The clean ResNet50 path is frozen. One automatic unnamed-region branch reads
    either layer2 (ENTO-like) or layer3 (MVL-like). The branch is independently
    class-supervised and is added to the frozen clean logits through a learnable
    scalar initialized at zero. Therefore epoch 0 is exactly the supplied clean
    ResNet50 prediction.

    No CUB attribute target, part label, or bounding box is used.
    """

    def __init__(
        self,
        num_classes: int,
        region: str,
        num_regions: int = 4,
        region_dim: int = 256,
        branch_dim: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if region not in {"ento", "mvl"}:
            raise ValueError("region must be 'ento' or 'mvl'")
        from models import CleanResNet50

        self.region = region
        self.backbone = CleanResNet50(
            num_classes=num_classes,
            pretrained=False,
        )
        # ENTO-like reads ResNet50 layer2 output f2 (512 channels).
        # MVL-like in this generic single-region ablation reads layer3 f3 (1024 channels).
        in_channels = 512 if region == "ento" else 1024
        self.region_encoder = AutomaticRegionEncoder(
            in_channels=in_channels,
            num_regions=num_regions,
            region_dim=region_dim,
            output_dim=branch_dim,
            dropout=dropout,
        )
        self.branch_classifier = nn.Linear(branch_dim, num_classes)
        nn.init.normal_(self.branch_classifier.weight, std=0.01)
        nn.init.zeros_(self.branch_classifier.bias)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def load_clean(self, checkpoint_path: str, prefer_ema: bool = False):
        from models import load_checkpoint_state

        _, state = load_checkpoint_state(
            checkpoint_path,
            prefer_ema=prefer_ema,
        )
        current = self.backbone.state_dict()
        transferable = {
            key: value
            for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        result = self.backbone.load_state_dict(transferable, strict=False)
        return result

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        for parameter in self.region_encoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.branch_classifier.parameters():
            parameter.requires_grad_(True)
        self.alpha.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _feature_map(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.region == "ento":
            # Canonical ENTO-like input: ResNet50 layer2 feature map f2.
            return features["f2"]
        return features["f3"]

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone is frozen, so its activations need no gradient graph.
        with torch.no_grad():
            features = self.backbone.forward_features(images)
            f4 = features["f4"]
            z_base = self.backbone.pool(f4).flatten(1)
            logits_base = self.backbone.classifier(z_base)
            feature_map = self._feature_map(features).detach()

        region_output = self.region_encoder(feature_map)
        embedding = region_output["embedding"]
        logits_branch = self.branch_classifier(embedding)
        logits = logits_base + self.alpha * logits_branch
        return {
            "logits": logits,
            "logits_base": logits_base,
            "logits_branch": logits_branch,
            "embedding": embedding,
            "heatmaps": region_output["heatmaps"],
            "region_weights": region_output["region_weights"],
            "alpha": self.alpha,
        }
