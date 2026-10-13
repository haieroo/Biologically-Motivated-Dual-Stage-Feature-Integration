from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import CleanResNet50, load_checkpoint_state


VALID_MVL_MODES = {
    "full_mvl",
    "no_foreground_guide",
    "single_scale",
    "no_covariance",
    "compact_mvl",
    "compact_no_covariance",
}


def mode_components(mode: str) -> Dict[str, bool]:
    if mode not in VALID_MVL_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALID_MVL_MODES)}"
        )
    return {
        "foreground_guide": mode not in {
            "no_foreground_guide",
            "compact_mvl",
            "compact_no_covariance",
        },
        "distinct_multiscale_dilations": mode not in {
            "single_scale",
            "compact_mvl",
            "compact_no_covariance",
        },
        "second_order_covariance": mode not in {
            "no_covariance",
            "compact_no_covariance",
        },
    }


class CompactMVLPlugin(nn.Module):
    """
    Matched MVL-like local-detail module.

    full_mvl:
        Foreground guidance + three dilation scales + covariance.

    no_foreground_guide:
        Three scales + covariance with uniform spatial statistics.

    single_scale:
        Foreground guidance + one local 3x3 scale + covariance.

    no_covariance:
        Foreground guidance + three scales without covariance.

    compact_mvl:
        One local 3x3 scale + uniform spatial statistics + covariance.

    compact_no_covariance:
        One local 3x3 scale + uniform spatial statistics without covariance.

    Disabled components are bypassed while the declared output dimensions stay
    matched across modes.
    """

    def __init__(
        self,
        mode: str,
        layer3_channels: int = 1024,
        layer4_channels: int = 2048,
        hidden_channels: int = 256,
        covariance_channels: int = 64,
        embed_dim: int = 512,
        num_classes: int = 200,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mode = mode
        self.components = mode_components(mode)

        self.foreground_guide = nn.Sequential(
            nn.Conv2d(layer4_channels, 256, kernel_size=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            nn.Conv2d(256, 1, kernel_size=1),
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(
                layer3_channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(32, hidden_channels),
            nn.GELU(),
        )
        self.texture_d1 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(32, hidden_channels),
            nn.GELU(),
        )
        self.texture_d2 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(32, hidden_channels),
            nn.GELU(),
        )
        self.texture_d3 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(32, hidden_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 3,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(32, hidden_channels),
            nn.GELU(),
        )

        self.first_order_project = nn.Sequential(
            nn.Linear(hidden_channels * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.covariance_reduce = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                covariance_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, covariance_channels),
            nn.GELU(),
        )
        upper_dim = covariance_channels * (covariance_channels + 1) // 2
        self.second_order_project = nn.Sequential(
            nn.Linear(upper_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_project = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        triangle = torch.triu_indices(
            covariance_channels,
            covariance_channels,
        )
        self.register_buffer(
            "tri_rows",
            triangle[0],
            persistent=False,
        )
        self.register_buffer(
            "tri_cols",
            triangle[1],
            persistent=False,
        )

    @staticmethod
    def _weighted_mean_std(
        feature: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = weight / weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1e-6)
        mean = (feature * weight).sum(dim=(2, 3))
        centered = feature - mean[:, :, None, None]
        variance = (centered.square() * weight).sum(dim=(2, 3))
        std = torch.sqrt(
            variance.float() + 1e-5
        ).to(feature.dtype)
        return mean, std

    def _weighted_covariance(
        self,
        feature: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        x = feature.float()
        batch, channels, height, width = x.shape
        x = x.reshape(batch, channels, height * width)

        flat_weight = weight.float().reshape(
            batch,
            1,
            height * width,
        )
        flat_weight = flat_weight / flat_weight.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(1e-6)

        mean = (x * flat_weight).sum(
            dim=2,
            keepdim=True,
        )
        centered = x - mean
        covariance = torch.bmm(
            centered * flat_weight,
            centered.transpose(1, 2),
        )
        scale = covariance.diagonal(
            dim1=1,
            dim2=2,
        ).mean(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-5)
        covariance = covariance / scale[:, :, None]
        upper = covariance[
            :,
            self.tri_rows,
            self.tri_cols,
        ]
        return upper.to(feature.dtype)

    def _spatial_weight(
        self,
        f3: torch.Tensor,
        f4: torch.Tensor,
    ) -> torch.Tensor:
        if not self.components["foreground_guide"]:
            return torch.ones(
                f3.size(0),
                1,
                f3.size(2),
                f3.size(3),
                device=f3.device,
                dtype=f3.dtype,
            )

        guide = self.foreground_guide(f4)
        guide = F.interpolate(
            guide,
            size=f3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(guide)

    def forward(
        self,
        f3: torch.Tensor,
        f4: torch.Tensor,
    ):
        spatial_weight = self._spatial_weight(f3, f4)

        reduced = self.reduce(f3)
        texture_1 = self.texture_d1(reduced)

        if self.components["distinct_multiscale_dilations"]:
            texture_2 = self.texture_d2(reduced)
            texture_3 = self.texture_d3(reduced)
        else:
            texture_2 = texture_1
            texture_3 = texture_1

        texture = self.fuse(
            torch.cat(
                [texture_1, texture_2, texture_3],
                dim=1,
            )
        )

        if self.components["foreground_guide"]:
            texture = texture * (1.0 + spatial_weight)

        mean, std = self._weighted_mean_std(
            texture,
            spatial_weight,
        )
        first_order = self.first_order_project(
            torch.cat([mean, std], dim=1)
        )

        if self.components["second_order_covariance"]:
            covariance_feature = self.covariance_reduce(texture)
            covariance = self._weighted_covariance(
                covariance_feature,
                spatial_weight,
            )
            second_order = self.second_order_project(covariance)
        else:
            second_order = torch.zeros_like(first_order)

        embedding = self.output_project(
            torch.cat(
                [first_order, second_order],
                dim=1,
            )
        )
        logits = self.classifier(embedding)
        return embedding, logits, spatial_weight


class CompactMVLResNet50(nn.Module):
    """
    Clean ResNet50 plus a residual MVL-like local-detail plugin.

        final_logits = clean_logits + alpha_mvl * mvl_logits

    alpha_mvl is initialized to zero, so every mode starts exactly from the
    same Clean ResNet50 prediction.
    """

    def __init__(
        self,
        num_classes: int,
        mode: str,
        pretrained: bool = False,
        embed_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        base = CleanResNet50(
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.mode = mode
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool = base.pool
        self.classifier = base.classifier

        self.mvl_plugin = CompactMVLPlugin(
            mode=mode,
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.alpha_mvl = nn.Parameter(torch.tensor(0.0))

    def forward_to_layer2(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        return self.layer2(f1)

    def forward_features(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        f2 = self.forward_to_layer2(images)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {
            "f2": f2,
            "f3": f3,
            "f4": f4,
        }

    def logits_from_f3(
        self,
        f3: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        f4 = self.layer4(f3)
        z_base = self.pool(f4).flatten(1)
        logits_base = self.classifier(z_base)
        z_mvl, logits_mvl, spatial_weight = self.mvl_plugin(f3, f4)
        logits = logits_base + self.alpha_mvl * logits_mvl
        return {
            "logits": logits,
            "logits_base": logits_base,
            "logits_mvl": logits_mvl,
            "z_base": z_base,
            "z_mvl": z_mvl,
            "spatial_weight": spatial_weight,
            "f4": f4,
        }

    def forward(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        features = self.forward_features(images)
        final = self.logits_from_f3(features["f3"])
        return {
            **final,
            **features,
        }


def load_clean_checkpoint(
    model: CompactMVLResNet50,
    checkpoint_path: str,
    prefer_ema: bool = True,
):
    _, state = load_checkpoint_state(
        checkpoint_path,
        prefer_ema=prefer_ema,
    )
    model_state = model.state_dict()
    transferable = {
        key: value
        for key, value in state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    required_prefixes = (
        "conv1",
        "bn1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "classifier",
    )
    missing_required = [
        key
        for key in model_state
        if key.startswith(required_prefixes) and key not in transferable
    ]
    if missing_required:
        raise RuntimeError(
            "Clean checkpoint is incompatible. Missing examples: "
            f"{missing_required[:20]}"
        )
    return model.load_state_dict(transferable, strict=False)


def load_compact_mvl_checkpoint(
    model: CompactMVLResNet50,
    checkpoint_path: str,
    prefer_ema: bool = True,
):
    _, state = load_checkpoint_state(
        checkpoint_path,
        prefer_ema=prefer_ema,
    )
    model_state = model.state_dict()
    transferable = {
        key: value
        for key, value in state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing = [
        key
        for key in model_state
        if key not in transferable
        and not key.startswith("mvl_plugin.tri_")
    ]
    if missing:
        raise RuntimeError(
            "MVL checkpoint is incompatible. Missing examples: "
            f"{missing[:20]}"
        )
    return model.load_state_dict(transferable, strict=False)
