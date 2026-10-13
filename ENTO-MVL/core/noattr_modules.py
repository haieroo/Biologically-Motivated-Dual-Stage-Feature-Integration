from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutomaticRegionEncoder(nn.Module):
    """Learn unnamed soft spatial regions without part/attribute/bbox labels."""

    def __init__(
        self,
        in_channels: int,
        num_regions: int = 4,
        region_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_regions = int(num_regions)
        self.heatmap_head = nn.Conv2d(in_channels, self.num_regions, 1, bias=True)
        nn.init.normal_(self.heatmap_head.weight, std=0.01)
        nn.init.zeros_(self.heatmap_head.bias)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(in_channels), nn.Linear(in_channels, region_dim), nn.GELU()
        )
        self.region_score = nn.Sequential(nn.LayerNorm(region_dim), nn.Linear(region_dim, 1))
        self.output_projection = nn.Sequential(
            nn.LayerNorm(region_dim),
            nn.Linear(region_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, _, height, width = feature_map.shape
        heatmap_logits = self.heatmap_head(feature_map)
        heatmaps = F.softmax(heatmap_logits.flatten(2), dim=-1).view(
            batch, self.num_regions, height, width
        )
        feature_flat = feature_map.flatten(2).transpose(1, 2)
        tokens = torch.einsum("brn,bnc->brc", heatmaps.flatten(2), feature_flat)
        tokens = self.token_projection(tokens)
        region_weights = F.softmax(self.region_score(tokens).squeeze(-1), dim=1)
        pooled = torch.sum(region_weights.unsqueeze(-1) * tokens, dim=1)
        return {
            "embedding": self.output_projection(pooled),
            "tokens": tokens,
            "heatmaps": heatmaps,
            "heatmap_logits": heatmap_logits,
            "region_weights": region_weights,
        }


class BoundedResidualHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        logit_clip: float = 1.0,
    ) -> None:
        super().__init__()
        self.logit_clip = float(logit_clip)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features)
        if self.logit_clip <= 0:
            return raw
        return self.logit_clip * torch.tanh(raw / self.logit_clip)


def region_overlap_loss(heatmaps: torch.Tensor) -> torch.Tensor:
    """Penalize collapse of all unnamed regions onto the same location."""
    flattened = heatmaps.flatten(2)
    normalized = F.normalize(flattened, dim=2, eps=1e-8)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    regions = similarity.size(1)
    if regions <= 1:
        return similarity.sum() * 0.0
    off_diagonal = similarity - torch.eye(
        regions, device=similarity.device, dtype=similarity.dtype
    ).unsqueeze(0)
    return off_diagonal.square().sum() / (
        similarity.size(0) * regions * (regions - 1)
    )
