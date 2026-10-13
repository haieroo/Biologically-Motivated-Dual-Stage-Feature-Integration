from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
forbidden = [
    "E:\\apyy\\",
    "D:\\Anaconda3\\",
    "G:\\CUB_200_2011\\",
    "/root/autodl",
    "ASYM_ENTO_MVL_RAW_BATCH4_V3",
    "ento_mvl_second_model_v2_direct_code",
]
text_suffixes = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml"}
bad = []
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in text_suffixes:
        continue
    if "__pycache__" in p.parts or p.name == "test_public_release.py":
        continue
    text = p.read_text(encoding="utf-8-sig", errors="ignore")
    hits = [x for x in forbidden if x in text]
    if hits:
        bad.append((str(p.relative_to(ROOT)), hits))
if bad:
    raise RuntimeError(f"Private machine paths remain: {bad}")
print("PUBLIC RELEASE AUDIT PASS")
