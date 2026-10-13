from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn as nn

from compact_ento_compat import CompactStrongMVLWithENTOAGCL
from noattr_modules import AutomaticRegionEncoder, BoundedResidualHead


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class ResidualEmbeddingAdapter(nn.Module):
    """Residual adapter used inside the MVL-like pathway."""

    def __init__(
        self,
        dim: int = 512,
        bottleneck: int = 256,
        dropout: float = 0.10,
        residual_scale: float = 0.20,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding + self.residual_scale * self.net(embedding)


class UnifiedENTOMVLLike(nn.Module):
    """Shared ENTO-like/MVL-like implementation used by the paper model.

    The ENTO-like pathway reads the frozen ResNet-50 layer2 feature map and
    produces (i) standalone ENTO-like class scores and (ii) a bounded class-wise
    residual. The Compact-MVL source supplies the MVL-like output scores. The
    joint decision rule is defined over the complete class space; no candidate
    mask, re-ranking set, or class-selection operation is used.
    """

    ENTO_SOURCE_LAYER = "layer2"
    ENTO_SOURCE_FEATURE = "f2"
    ENTO_SOURCE_CHANNELS = 512

    def __init__(
        self,
        num_classes: int,
        num_attributes: int,
        neural_mode: str = "true_neural",
        embed_dim: int = 512,
        num_regions: int = 4,
        region_dim: int = 256,
        branch_dim: int = 512,
        mvl_adapter_bottleneck: int = 256,
        residual_hidden_dim: int = 256,
        residual_dropout: float = 0.10,
        residual_logit_clip: float = 1.0,
        residual_strength: float = 0.20,
        enable_ento: bool = True,
        enable_mvl_adapter: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_attributes = int(num_attributes)
        self.neural_mode = str(neural_mode)
        self.embed_dim = int(embed_dim)
        self.branch_dim = int(branch_dim)
        self.residual_strength = float(residual_strength)
        self.enable_ento = bool(enable_ento)
        self.enable_mvl_adapter = bool(enable_mvl_adapter)

        self.core = CompactStrongMVLWithENTOAGCL(
            num_classes=num_classes,
            num_attributes=num_attributes,
            embed_dim=embed_dim,
            projection_dim=128,
            dropout=0.20,
        )

        self.ento_regions = AutomaticRegionEncoder(
            in_channels=self.ENTO_SOURCE_CHANNELS,
            num_regions=num_regions,
            region_dim=region_dim,
            output_dim=branch_dim,
            dropout=residual_dropout,
        )
        self.ento_projection = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Linear(branch_dim, embed_dim),
            nn.GELU(),
        )
        self.mvl_adapter = ResidualEmbeddingAdapter(
            dim=embed_dim,
            bottleneck=mvl_adapter_bottleneck,
            dropout=residual_dropout,
            residual_scale=0.20,
        )
        self.ento_classifier = nn.Linear(embed_dim, num_classes)
        self.ento_residual = BoundedResidualHead(
            input_dim=embed_dim,
            num_classes=num_classes,
            hidden_dim=residual_hidden_dim,
            dropout=residual_dropout,
            logit_clip=residual_logit_clip,
        )

    def load_compact_mvl(self, checkpoint_path: str, prefer_ema: bool = True):
        return self.core.load_strong_mvl(checkpoint_path, prefer_ema=prefer_ema)

    def freeze_source(self) -> None:
        set_requires_grad(self.core, False)
        self.core.eval()
        for module in self.trainable_modules():
            set_requires_grad(module, True)

    def trainable_modules(self):
        modules = []
        if self.enable_ento:
            modules += [
                self.ento_regions,
                self.ento_projection,
                self.ento_classifier,
                self.ento_residual,
            ]
        if self.enable_mvl_adapter:
            modules += [self.mvl_adapter]
        return tuple(modules)

    def trainable_parameters(self):
        return [
            parameter
            for module in self.trainable_modules()
            for parameter in module.parameters()
        ]

    def train(self, mode: bool = True):
        super().train(mode)
        # The trained MVL-like source is fixed during final ENTO adaptation.
        self.core.eval()
        return self

    @torch.no_grad()
    def extract_frozen_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.core.eval()
        output = self.core(images, compute_ento=False)
        return {
            "f2": output["f2"].detach(),
            "source_logits": output["logits"].detach(),
            "z_mvl_source": output["z_mvl"].detach(),
        }

    def branch_forward_from_features(
        self, frozen: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch = frozen["source_logits"].size(0)
        dtype = frozen["z_mvl_source"].dtype
        device = frozen["z_mvl_source"].device

        if self.enable_ento:
            region = self.ento_regions(frozen["f2"])
            z_ento = self.ento_projection(region["embedding"])
            ento_logits = self.ento_classifier(z_ento)
            ento_delta = self.ento_residual(z_ento)
            heatmaps = region["heatmaps"]
            region_weights = region["region_weights"]
        else:
            z_ento = torch.zeros(batch, self.embed_dim, device=device, dtype=dtype)
            ento_logits = torch.zeros(
                batch, self.num_classes, device=device, dtype=dtype
            )
            ento_delta = torch.zeros_like(ento_logits)
            heatmaps = torch.empty(0, device=device, dtype=dtype)
            region_weights = torch.empty(0, device=device, dtype=dtype)

        z_mvl = (
            self.mvl_adapter(frozen["z_mvl_source"])
            if self.enable_mvl_adapter
            else frozen["z_mvl_source"]
        )

        return {
            "z_ento": z_ento,
            "z_mvl": z_mvl,
            "z_mvl_source": frozen["z_mvl_source"],
            "ento_logits": ento_logits,
            "ento_delta": ento_delta,
            "ento_heatmaps": heatmaps,
            "ento_region_weights": region_weights,
        }

    def apply_all_class_residual(
        self,
        source_logits: torch.Tensor,
        ento_delta: torch.Tensor,
        scale_factor: Union[torch.Tensor, float] = 1.0,
        use_ento: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Apply the ENTO-like residual to every class score.

        z_A = M + alpha * s_E * Delta_E

        `source_logits` is the complete MVL-like output M supplied by the
        frozen Compact-MVL source checkpoint.
        """
        total_delta = (
            ento_delta
            if self.enable_ento and use_ento
            else torch.zeros_like(source_logits)
        )
        scale = torch.as_tensor(
            scale_factor,
            device=source_logits.device,
            dtype=source_logits.dtype,
        )
        scaled_delta = scale * total_delta
        final_logits = source_logits + self.residual_strength * scaled_delta
        prediction = final_logits.argmax(dim=1)
        return {
            "prediction": prediction,
            "total_delta": total_delta,
            "scaled_delta": scaled_delta,
            "final_logits": final_logits,
        }

    def forward_from_features(
        self, frozen: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        branch = self.branch_forward_from_features(frozen)
        full = self.apply_all_class_residual(
            frozen["source_logits"], branch["ento_delta"]
        )
        source_only = self.apply_all_class_residual(
            frozen["source_logits"], branch["ento_delta"], use_ento=False
        )
        return {
            **frozen,
            **branch,
            "full_prediction": full["prediction"],
            "source_only_prediction": source_only["prediction"],
            "residual_delta": full["total_delta"],
            "scaled_residual_delta": full["scaled_delta"],
            "final_logits": full["final_logits"],
        }

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_from_features(self.extract_frozen_features(images))
