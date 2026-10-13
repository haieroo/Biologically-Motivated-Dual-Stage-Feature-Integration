from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Iterable, List, Optional

MAX_SEED = 2**31 - 1

def random_seed() -> int:
    return secrets.randbelow(MAX_SEED - 1) + 1

def resolve_seed(seed: Optional[int], label: str = "run") -> int:
    value = random_seed() if seed is None else int(seed)
    if not (0 < value < MAX_SEED):
        raise ValueError(f"{label} seed out of range: {value}")
    print(
        f"[SEED] {label}: "
        + (f"auto-generated {value}" if seed is None else f"user-specified {value}")
    )
    return value

def generate_unique_seeds(n: int) -> List[int]:
    if n < 1:
        raise ValueError("num-runs must be >= 1")
    out = []
    seen = set()
    while len(out) < n:
        s = random_seed()
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def parse_seed_list(text: str) -> List[int]:
    text = (text or "").strip()
    if not text:
        return []
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != len(set(values)):
        raise ValueError("Duplicate seeds are not allowed")
    return values

def save_manifest(path: Path, seeds: Iterable[int], **extra) -> None:
    payload = {
        "seeds": [int(x) for x in seeds],
        "seed_policy": "OS-random by default; saved for reproducibility",
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
