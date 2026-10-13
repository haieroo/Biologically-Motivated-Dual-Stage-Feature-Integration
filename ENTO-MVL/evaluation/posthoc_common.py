# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata, spearmanr
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
CORE = REPO_ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from models_ento_mvl import ENTOMVL
from neural_rdm_utils import load_neural_manifest, load_square_matrix

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_historical_amp() -> None:
    # Canonical evaluation path established by the unified source/joint audit.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def autocast_context(enabled: bool):
    if torch.cuda.is_available():
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_joint_model(checkpoint_path: Path, device: torch.device):
    ck = safe_torch_load(checkpoint_path)
    args = dict(ck.get("args", {}))
    classes = ck.get("classes") or []
    attribute_names = ck.get("attribute_names") or []

    model = ENTOMVL(
        num_classes=len(classes) or 200,
        num_attributes=len(attribute_names),
        neural_mode="no_neural",
        embed_dim=512,
        num_regions=int(args.get("num_regions", 4)),
        region_dim=int(args.get("region_dim", 256)),
        branch_dim=int(args.get("branch_dim", 512)),
        mvl_adapter_bottleneck=256,
        residual_hidden_dim=int(args.get("residual_hidden_dim", 256)),
        residual_dropout=float(args.get("residual_dropout", 0.10)),
        residual_logit_clip=float(args.get("residual_logit_clip", 1.0)),
        residual_strength=float(ck.get("alpha_E", args.get("alpha_e", 0.20))),
        ento_scale_factor=float(ck.get("s_E", 1.0)),
    )
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def eval_transform(img_size: int = 448):
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class OrderedStimulusDataset(Dataset):
    def __init__(
        self,
        manifest_rows: Sequence[Mapping[str, str]],
        img_size: int = 448,
        image_transform=None,
    ):
        self.rows = list(manifest_rows)
        self.tf = image_transform if image_transform is not None else eval_transform(img_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        p = Path(self.rows[idx]["image_path"])
        with Image.open(p) as im:
            x = self.tf(im.convert("RGB"))
        return x, idx, self.rows[idx]["stimulus_id"], str(p)


def upper_vector(rdm: np.ndarray) -> np.ndarray:
    rdm = np.asarray(rdm, dtype=np.float64)
    i, j = np.triu_indices(rdm.shape[0], 1)
    return rdm[i, j]


def pearson_distance_rdm(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    norm = np.maximum(norm, 1e-12)
    z = x / norm
    corr = np.clip(z @ z.T, -1.0, 1.0)
    rdm = 1.0 - corr
    rdm = 0.5 * (rdm + rdm.T)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def spearman_rdm(a: np.ndarray, b: np.ndarray) -> float:
    av = upper_vector(a)
    bv = upper_vector(b)
    mask = np.isfinite(av) & np.isfinite(bv)
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(av[mask], bv[mask]).statistic)


def permute_rdm_labels(rdm: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return np.asarray(rdm)[np.ix_(permutation, permutation)]


def stimulus_label_permutation_test(
    model_rdm: np.ndarray,
    target_rdm: np.ndarray,
    permutations: int,
    seed: int,
) -> Dict[str, float]:
    obs = spearman_rdm(model_rdm, target_rdm)
    rng = np.random.default_rng(int(seed))
    count_two = 0
    n = model_rdm.shape[0]
    for _ in range(int(permutations)):
        p = rng.permutation(n)
        rho = spearman_rdm(model_rdm, permute_rdm_labels(target_rdm, p))
        if abs(rho) >= abs(obs) - 1e-15:
            count_two += 1
    return {
        "rho": float(obs),
        "p_perm_two_sided": float((count_two + 1) / (int(permutations) + 1)),
        "permutations": int(permutations),
    }


def stimulus_label_delta_test(
    first_rdm: np.ndarray,
    second_rdm: np.ndarray,
    target_rdm: np.ndarray,
    permutations: int,
    seed: int,
) -> Dict[str, float]:
    rho_first = spearman_rdm(first_rdm, target_rdm)
    rho_second = spearman_rdm(second_rdm, target_rdm)
    obs = rho_first - rho_second
    rng = np.random.default_rng(int(seed))
    n = target_rdm.shape[0]
    count_two = 0
    for _ in range(int(permutations)):
        p = rng.permutation(n)
        t = permute_rdm_labels(target_rdm, p)
        delta = spearman_rdm(first_rdm, t) - spearman_rdm(second_rdm, t)
        if abs(delta) >= abs(obs) - 1e-15:
            count_two += 1
    return {
        "rho_first": float(rho_first),
        "rho_second": float(rho_second),
        "delta_rho": float(obs),
        "p_perm_two_sided": float((count_two + 1) / (int(permutations) + 1)),
        "permutations": int(permutations),
    }


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return out.tolist()
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    restored = np.empty_like(adj)
    restored[order] = adj
    out[valid] = restored
    return out.tolist()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def json_safe(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and (not np.isfinite(x)):
        return None
    if isinstance(x, Mapping):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_seed(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32)
