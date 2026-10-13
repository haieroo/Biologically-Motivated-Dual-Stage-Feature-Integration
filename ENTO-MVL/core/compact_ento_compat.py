from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models_compact_mvl import CompactMVLResNet50, load_compact_mvl_checkpoint


class CompactStrongMVLWithENTOAGCL(nn.Module):
    """Compatibility name for the frozen Compact-MVL source used by this package.

    ENTO is implemented only by the separate unnamed-region encoder in
    ``models_unified_mvllike.py``. This source wrapper contains no ENTO readout
    and no cross-path operation.
    """

    def __init__(
        self,
        num_classes: int,
        num_attributes: int,
        embed_dim: int = 512,
        projection_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        del num_attributes, projection_dim
        self.anchor = CompactMVLResNet50(
            num_classes=num_classes,
            mode="compact_mvl",
            pretrained=False,
            embed_dim=embed_dim,
            dropout=dropout,
        )

    def load_strong_mvl(self, checkpoint_path: str, prefer_ema: bool = True):
        return load_compact_mvl_checkpoint(
            self.anchor, checkpoint_path, prefer_ema=prefer_ema
        )

    def forward(self, images: torch.Tensor, compute_ento: bool = False) -> Dict[str, torch.Tensor]:
        if compute_ento:
            raise ValueError("ENTO is a separate branch; the Compact-MVL source cannot compute it")
        features = self.anchor.forward_features(images)
        f3 = features["f3"]
        f4 = features["f4"]
        z_base = self.anchor.pool(f4).flatten(1)
        logits_base = self.anchor.classifier(z_base)
        z_mvl, logits_mvl, spatial_weight = self.anchor.mvl_plugin(f3, f4)
        logits = logits_base + self.anchor.alpha_mvl * logits_mvl
        return {
            "logits": logits,
            "logits_base": logits_base,
            "logits_mvl": logits_mvl,
            "z_base": z_base,
            "z_mvl": z_mvl,
            "attention": spatial_weight,
            "spatial_weight": spatial_weight,
            **features,
        }
