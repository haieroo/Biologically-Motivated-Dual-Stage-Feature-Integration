from __future__ import annotations

from typing import Dict

import torch

from models_unified_mvllike import UnifiedENTOMVLLike


class ENTOMVL(UnifiedENTOMVLLike):
    """Final all-class ENTO–MVL classifier used in the revised manuscript.

    Final decision rule:
        M = z_base + alpha_M * z_M        (provided by the trained MVL source)
        z_A = M + alpha * s_E * Delta_E  (applied to all classes)

    There is no class-candidate mask, no joint interaction head, and no extra
    MVL residual added at the final integration step. `s_E` is a fixed,
    non-trainable RMS calibration factor fitted using training images only.
    """

    def __init__(self, *args, ento_scale_factor: float = 1.0, **kwargs):
        kwargs["enable_ento"] = True
        kwargs["enable_mvl_adapter"] = False
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "ento_scale_factor",
            torch.tensor(float(ento_scale_factor), dtype=torch.float32),
        )

    @torch.no_grad()
    def set_ento_scale_factor(self, value: float) -> None:
        value = float(value)
        if not (value > 0.0):
            raise ValueError(f"s_E must be positive, got {value}")
        self.ento_scale_factor.fill_(value)

    def forward_from_features(
        self, frozen: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        branch = self.branch_forward_from_features(frozen)
        full = self.apply_all_class_residual(
            frozen["source_logits"],
            branch["ento_delta"],
            scale_factor=self.ento_scale_factor,
        )
        source_only = self.apply_all_class_residual(
            frozen["source_logits"],
            branch["ento_delta"],
            scale_factor=self.ento_scale_factor,
            use_ento=False,
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
