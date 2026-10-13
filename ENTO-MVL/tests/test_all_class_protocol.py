from pathlib import Path
import sys
import torch

R = Path(__file__).resolve().parents[1]
CORE = R / "core"
sys.path.insert(0, str(CORE))

from models_ento_mvl import ENTOMVL

trainer = (R / "training" / "train_ento_mvl.py").read_text(encoding="utf-8-sig")
model_source = (R / "core" / "models_unified_mvllike.py").read_text(encoding="utf-8-sig")
assert "apply_all_class_residual" in model_source

C = 7
model = ENTOMVL(
    num_classes=C,
    num_attributes=1,
    neural_mode="no_neural",
    embed_dim=512,
    num_regions=4,
    region_dim=256,
    branch_dim=512,
    residual_hidden_dim=32,
    residual_dropout=0.0,
    residual_logit_clip=1.0,
    residual_strength=0.2,
    ento_scale_factor=1.7,
)
source = torch.randn(4, C)
delta = torch.randn(4, C)
out = model.apply_all_class_residual(source, delta, scale_factor=model.ento_scale_factor)
expected = source + 0.2 * 1.7 * delta
torch.testing.assert_close(out["final_logits"], expected, rtol=1e-6, atol=1e-6)
assert torch.equal(out["prediction"], expected.argmax(dim=1))
print("ALL-CLASS PROTOCOL TEST PASS")
