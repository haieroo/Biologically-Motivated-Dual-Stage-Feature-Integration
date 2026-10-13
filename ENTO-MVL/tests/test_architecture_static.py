from pathlib import Path

R = Path(__file__).resolve().parents[1]
s = (R / "core" / "models_single_region_branch.py").read_text(encoding="utf-8-sig")
u = (R / "core" / "models_unified_mvllike.py").read_text(encoding="utf-8-sig")
f = (R / "core" / "models_ento_mvl.py").read_text(encoding="utf-8-sig")

assert 'ENTO_SOURCE_LAYER = "layer2"' in s
assert 'return features["f2"]' in s
assert 'in_channels = 512 if region == "ento" else 1024' in s
assert 'ENTO_SOURCE_LAYER = "layer2"' in u
assert 'self.ENTO_SOURCE_CHANNELS' in u
assert 'region = self.ento_regions(frozen["f2"])' in u
assert 'kwargs["enable_mvl_adapter"] = False' in f
assert "interaction_delta" not in f
assert "mvl_delta" not in f
assert "apply_all_class_residual" in u
print("ARCHITECTURE TEST PASS")
print("ENTO source = ResNet50 layer2 f2; final residual is applied to all classes")
