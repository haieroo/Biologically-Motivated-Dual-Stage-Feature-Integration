from pathlib import Path

R = Path(__file__).resolve().parents[1]
runner = (R / "scripts" / "run_experiments.py").read_text(encoding="utf-8")
assert "666,2026,7777" in runner
assert "PAPER_SEEDS = [666, 2026, 7777]" in runner
for p in [
    R / "training" / "train_mvl_like.py",
    R / "training" / "train_ento_like.py",
    R / "training" / "train_ento_mvl.py",
]:
    t = p.read_text(encoding="utf-8-sig")
    seed_lines = [line for line in t.splitlines() if '"--seed"' in line]
    assert seed_lines
print("SEED POLICY TEST PASS")
print("Default manuscript seeds = 666, 2026, 7777")
