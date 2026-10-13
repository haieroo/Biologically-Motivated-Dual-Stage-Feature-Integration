# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import io
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from posthoc_common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OrderedStimulusDataset,
    autocast_context,
    build_joint_model,
    configure_historical_amp,
    load_neural_manifest,
    load_square_matrix,
    pearson_distance_rdm,
    set_seed,
    sha256_seed,
    spearman_rdm,
    write_csv,
    write_json,
)

# Explicit, locked current-posthoc core protocol.
CORE = OrderedDict([
    ("occlusion", [0.20, 0.40, 0.60]),
    ("patch_shuffle", [0.33, 0.66, 1.00]),
    ("gaussian_blur", [1.50, 3.00, 4.50]),
    ("gaussian_noise", [0.08, 0.16, 0.24]),
    ("color_shift", [0.15, 0.30, 0.45]),
])
QUICK = OrderedDict([
    ("occlusion", [0.40]),
    ("patch_shuffle", [0.66]),
    ("gaussian_blur", [3.00]),
    ("gaussian_noise", [0.16]),
    ("color_shift", [0.30]),
])


def parse_args():
    p = argparse.ArgumentParser(
        "Final 3-seed classification + representational robustness evaluator."
    )
    p.add_argument("--project-root", required=True)
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--seeds", default="", help="Optional. If omitted, auto-discover results/seed* runs.")
    p.add_argument("--profile", choices=["quick", "core"], default="core")
    p.add_argument("--corruption-seed", type=int, default=20260810)
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--max-test-images", type=int, default=0,
        help="0 = all official evaluation images; positive integer = deterministic prefix for quick diagnostics."
    )
    p.add_argument("--manifest", default="")
    p.add_argument("--behavior-rdm", default="")
    p.add_argument("--ento-neural-rdm", default="")
    p.add_argument("--mvl-neural-rdm", default="")
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def condition_name(kind: str, severity: Optional[float]) -> str:
    if kind == "clean":
        return "clean"
    return f"{kind}_{float(severity):.3f}"


def rng_for(identity: str, kind: str, severity: float, corruption_seed: int):
    return np.random.default_rng(
        sha256_seed(corruption_seed, kind, f"{severity:.6f}", identity)
    )


def apply_corruption(
    img: Image.Image,
    kind: str,
    severity: float,
    identity: str,
    corruption_seed: int,
) -> Image.Image:
    """
    Locked CURRENT posthoc corruption protocol.

    NOTE: It intentionally uses the same five families/level labels as the
    user's prior robustness figures, but this implementation is explicitly
    documented here and should not be described as bitwise-identical to an
    older robustness script unless separately audited.
    """
    img = img.convert("RGB")
    s = float(severity)
    rng = rng_for(identity, kind, s, corruption_seed)

    if kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=s))

    if kind == "gaussian_noise":
        a = np.asarray(img).astype(np.float32) / 255.0
        a += rng.normal(0.0, s, size=a.shape).astype(np.float32)
        a = np.clip(a, 0.0, 1.0)
        return Image.fromarray(np.round(a * 255.0).astype(np.uint8))

    if kind == "occlusion":
        # Severity is OCCLUDED AREA fraction. Use one deterministic square.
        a = np.asarray(img).copy()
        h, w = a.shape[:2]
        side_frac = math.sqrt(max(0.0, min(s, 0.95)))
        oh = max(1, min(h, int(round(h * side_frac))))
        ow = max(1, min(w, int(round(w * side_frac))))
        y0 = int(rng.integers(0, max(1, h - oh + 1)))
        x0 = int(rng.integers(0, max(1, w - ow + 1)))
        a[y0:y0 + oh, x0:x0 + ow] = np.asarray([123, 116, 103], dtype=np.uint8)
        return Image.fromarray(a)

    if kind == "patch_shuffle":
        # Fixed 4x4 grid. Severity is fraction of patch positions participating
        # in the permutation; 1.0 shuffles all patches.
        a = np.asarray(img).copy()
        h, w = a.shape[:2]
        grid = 4
        ys = np.linspace(0, h, grid + 1, dtype=int)
        xs = np.linspace(0, w, grid + 1, dtype=int)
        cells = [(iy, ix) for iy in range(grid) for ix in range(grid)]
        count = max(2, min(len(cells), int(round(len(cells) * s))))
        chosen_pos = rng.choice(len(cells), size=count, replace=False)
        chosen = [cells[int(i)] for i in chosen_pos]
        perm = rng.permutation(count)
        patches = []
        for iy, ix in chosen:
            patches.append(a[ys[iy]:ys[iy+1], xs[ix]:xs[ix+1]].copy())
        out = a.copy()
        for target_index, (iy, ix) in enumerate(chosen):
            source_patch = patches[int(perm[target_index])]
            th = ys[iy+1] - ys[iy]
            tw = xs[ix+1] - xs[ix]
            if source_patch.shape[0] != th or source_patch.shape[1] != tw:
                source_patch = np.asarray(
                    Image.fromarray(source_patch).resize((tw, th), Image.BILINEAR)
                )
            out[ys[iy]:ys[iy+1], xs[ix]:xs[ix+1]] = source_patch
        return Image.fromarray(out)

    if kind == "color_shift":
        # Mild deterministic hue rotation. severity=0.45 corresponds to
        # approximately 13 degrees hue rotation (0.45*0.08 of a full cycle).
        hsv = np.asarray(img.convert("HSV")).copy()
        hue_shift = int(round(255.0 * s * 0.08))
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 256
        return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")

    raise ValueError(kind)


class CorruptImageFolder(Dataset):
    def __init__(
        self,
        root: Path,
        img_size: int,
        kind: str,
        severity: Optional[float],
        corruption_seed: int,
        max_images: int = 0,
    ):
        base = datasets.ImageFolder(str(root))
        self.samples = base.samples
        self.classes = base.classes
        self.loader = base.loader
        self.kind = kind
        self.severity = severity
        self.corruption_seed = int(corruption_seed)
        if max_images > 0:
            self.samples = self.samples[: int(max_images)]
        self.resize = transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path).convert("RGB")
        img = self.resize(img)
        if self.kind != "clean":
            img = apply_corruption(
                img, self.kind, float(self.severity), path, self.corruption_seed
            )
        return self.norm(self.tensor(img)), target, path


class CorruptStimuli(Dataset):
    def __init__(
        self,
        manifest_rows,
        img_size,
        kind,
        severity,
        corruption_seed,
    ):
        self.rows = list(manifest_rows)
        self.kind = kind
        self.severity = severity
        self.corruption_seed = corruption_seed
        self.resize = transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        p = Path(row["image_path"])
        with Image.open(p) as im:
            img = self.resize(im.convert("RGB"))
        if self.kind != "clean":
            img = apply_corruption(
                img, self.kind, float(self.severity), str(p), self.corruption_seed
            )
        return self.norm(self.tensor(img)), idx, row["stimulus_id"], str(p)


@torch.no_grad()
def evaluate_classification(model, loader, device):
    n = src_correct = final_correct = src_top5 = final_top5 = 0
    corrected = degraded = changed = 0
    for images, labels, _paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device.type == "cuda"):
            out = model(images)
        src_logits = out["source_logits"]
        final_logits = out["final_logits"]
        sp = src_logits.argmax(1)
        fp = out["full_prediction"]
        sc = sp.eq(labels)
        fc = fp.eq(labels)
        n += labels.numel()
        src_correct += int(sc.sum().item())
        final_correct += int(fc.sum().item())
        changed += int(sp.ne(fp).sum().item())
        corrected += int(((~sc) & fc).sum().item())
        degraded += int((sc & (~fc)).sum().item())
        src_top5 += int(src_logits.topk(min(5, src_logits.size(1)), 1).indices.eq(labels[:, None]).any(1).sum().item())
        final_top5 += int(final_logits.topk(min(5, final_logits.size(1)), 1).indices.eq(labels[:, None]).any(1).sum().item())
    return {
        "n": n,
        "source_top1": 100.0 * src_correct / n,
        "final_top1": 100.0 * final_correct / n,
        "delta_top1_pp": 100.0 * (final_correct - src_correct) / n,
        "source_top5": 100.0 * src_top5 / n,
        "final_top5": 100.0 * final_top5 / n,
        "changed": changed,
        "corrected": corrected,
        "degraded": degraded,
        "net": corrected - degraded,
    }


@torch.no_grad()
def extract_stimulus_features(model, loader, device):
    pieces = {k: [] for k in [
        "z_ento", "z_mvl", "z_joint_mean", "source_logits", "final_logits"
    ]}
    order = []
    for images, idx, _sid, _path in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(device.type == "cuda"):
            out = model(images)
        ze = out["z_ento"].float()
        zm = out["z_mvl"].float()
        pieces["z_ento"].append(ze.cpu())
        pieces["z_mvl"].append(zm.cpu())
        pieces["z_joint_mean"].append((0.5 * (ze + zm)).cpu())
        pieces["source_logits"].append(out["source_logits"].float().cpu())
        pieces["final_logits"].append(out["final_logits"].float().cpu())
        order.extend(int(x) for x in idx)
    if order != list(range(len(order))):
        raise RuntimeError("Stimulus order changed.")
    return {k: torch.cat(v, 0).numpy() for k, v in pieces.items()}


def condition_grid(profile):
    spec = QUICK if profile == "quick" else CORE
    out = [("clean", None)]
    for kind, values in spec.items():
        for s in values:
            out.append((kind, float(s)))
    return out


def mean_sd(x):
    x = np.asarray(list(x), dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0



def discover_seeds(project_root: Path):
    result_root = Path(project_root) / "results"
    seeds = []
    if result_root.exists():
        for p in result_root.glob("seed*"):
            suffix = p.name[4:]
            if p.is_dir() and suffix.isdigit():
                seeds.append(int(suffix))
    if not seeds:
        raise RuntimeError(
            f"No completed seed directories found under {result_root}. "
            "Run scripts/run_experiments.py first or pass --seeds explicitly."
        )
    return sorted(seeds)


def main():
    a = parse_args()
    project = Path(a.project_root).resolve()
    legacy = Path(a.legacy_root).resolve()
    data_root = Path(a.data_root).resolve()
    eval_dir = data_root / "val"
    if not eval_dir.exists():
        alt = data_root / "test"
        if alt.exists():
            eval_dir = alt
        else:
            raise FileNotFoundError(f"No val/ or test/ under {data_root}")

    input_dir = legacy / "neural_rdm_inputs"
    manifest_path = Path(a.manifest) if a.manifest else input_dir / "36_neural_manifest.csv"
    behavior_path = Path(a.behavior_rdm) if a.behavior_rdm else input_dir / "Behavior_RDM.csv"
    ento_path = Path(a.ento_neural_rdm) if a.ento_neural_rdm else input_dir / "ENTO_crossnobis_RDM.csv"
    mvl_path = Path(a.mvl_neural_rdm) if a.mvl_neural_rdm else input_dir / "MVL_crossnobis_RDM.csv"
    outdir = (
        Path(a.output_dir).resolve()
        if a.output_dir
        else project / "results" / f"final_robustness_3seeds_{a.profile}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = load_neural_manifest(str(manifest_path))
    behavior = load_square_matrix(str(behavior_path))
    ento_neural = load_square_matrix(str(ento_path))
    mvl_neural = load_square_matrix(str(mvl_path))
    if not (behavior.shape == ento_neural.shape == mvl_neural.shape == (len(manifest), len(manifest))):
        raise RuntimeError("Manifest / behavior / neural RDM dimensions differ.")

    seeds = ([int(x.strip()) for x in a.seeds.split(",") if x.strip()] if str(a.seeds).strip() else discover_seeds(project))
    conditions = condition_grid(a.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_historical_amp()

    print("=" * 148)
    print("FINAL ENTO-MVL 3-SEED ROBUSTNESS")
    print(f"Seeds={seeds} | profile={a.profile} | conditions={len(conditions)}")
    print(f"CorruptionSeed={a.corruption_seed} (identical realization across model seeds)")
    print(f"Official eval={eval_dir} | MaxTestImages={a.max_test_images} (0=all)")
    print("Evaluation-only; no robustness result is used for training/checkpoint selection.")
    print("Current fixed perturbation implementation is documented in PROTOCOL.json.")
    print("=" * 148)

    all_rows: List[Dict] = []
    all_rdm_rows: List[Dict] = []

    for seed in seeds:
        set_seed(seed)
        ckpt = (
            project / "results" / f"seed{seed}"
            / "ento_mvl" / "final_ento_mvl.pth"
        )
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model, meta = build_joint_model(ckpt, device)

        clean_rdms = None
        print("\n" + "-" * 148)
        print(f"SEED {seed} | sE={float(meta.get('s_E', 1.0)):.6f}")
        print("-" * 148)

        for kind, sev in conditions:
            name = condition_name(kind, sev)
            cls_ds = CorruptImageFolder(
                eval_dir, a.img_size, kind, sev, a.corruption_seed, a.max_test_images
            )
            cls_loader = DataLoader(
                cls_ds,
                batch_size=a.batch_size,
                shuffle=False,
                num_workers=a.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=a.num_workers > 0,
            )
            cls = evaluate_classification(model, cls_loader, device)

            stim_ds = CorruptStimuli(
                manifest, a.img_size, kind, sev, a.corruption_seed
            )
            stim_loader = DataLoader(
                stim_ds,
                batch_size=min(12, max(1, a.batch_size * 2)),
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            feat = extract_stimulus_features(model, stim_loader, device)
            rdms = {k: pearson_distance_rdm(v) for k, v in feat.items()}
            if kind == "clean":
                clean_rdms = rdms
            assert clean_rdms is not None

            geometry = {
                "ento_retention": spearman_rdm(clean_rdms["z_ento"], rdms["z_ento"]),
                "mvl_retention": spearman_rdm(clean_rdms["z_mvl"], rdms["z_mvl"]),
                "joint_retention": spearman_rdm(clean_rdms["z_joint_mean"], rdms["z_joint_mean"]),
                "source_logits_retention": spearman_rdm(clean_rdms["source_logits"], rdms["source_logits"]),
                "final_logits_retention": spearman_rdm(clean_rdms["final_logits"], rdms["final_logits"]),
                "ento_to_ento_neural": spearman_rdm(rdms["z_ento"], ento_neural),
                "mvl_to_mvl_neural": spearman_rdm(rdms["z_mvl"], mvl_neural),
                "joint_to_ento_neural": spearman_rdm(rdms["z_joint_mean"], ento_neural),
                "joint_to_mvl_neural": spearman_rdm(rdms["z_joint_mean"], mvl_neural),
                "mvl_to_clean_behavior": spearman_rdm(rdms["z_mvl"], behavior),
                "joint_to_clean_behavior": spearman_rdm(rdms["z_joint_mean"], behavior),
            }

            row = {
                "seed": seed,
                "condition": name,
                "family": kind,
                "severity": "" if sev is None else sev,
                "sE": float(meta.get("s_E", 1.0)),
                **cls,
                **geometry,
            }
            all_rows.append(row)

            for rdm_name, matrix in rdms.items():
                np.save(
                    outdir / f"seed{seed}_{name}_{rdm_name}_RDM.npy",
                    matrix.astype(np.float32),
                )

            print(
                f"{name:28s} | "
                f"Src={cls['source_top1']:7.3f} Final={cls['final_top1']:7.3f} "
                f"Δ={cls['delta_top1_pp']:+.4f} | "
                f"JointRet={geometry['joint_retention']:+.4f} "
                f"E→E={geometry['ento_to_ento_neural']:+.4f} "
                f"M→M={geometry['mvl_to_mvl_neural']:+.4f} "
                f"J→Beh={geometry['joint_to_clean_behavior']:+.4f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(outdir / "01_per_seed_condition_results.csv", all_rows)

    # Family summaries over perturbed levels, first within seed then over seeds.
    per_seed_family = []
    for seed in seeds:
        seed_rows = [r for r in all_rows if r["seed"] == seed and r["family"] != "clean"]
        for family in CORE.keys():
            rr = [r for r in seed_rows if r["family"] == family]
            if not rr:
                continue
            for metric in [
                "source_top1", "final_top1", "delta_top1_pp",
                "ento_retention", "mvl_retention", "joint_retention",
                "source_logits_retention", "final_logits_retention",
                "ento_to_ento_neural", "mvl_to_mvl_neural",
                "joint_to_ento_neural", "joint_to_mvl_neural",
                "mvl_to_clean_behavior", "joint_to_clean_behavior",
            ]:
                per_seed_family.append({
                    "seed": seed,
                    "family": family,
                    "metric": metric,
                    "mean_across_severity": float(np.mean([float(x[metric]) for x in rr])),
                })
    write_csv(outdir / "02_per_seed_family_means.csv", per_seed_family)

    aggregate = []
    keys = sorted({(r["family"], r["metric"]) for r in per_seed_family})
    for family, metric in keys:
        vals = [
            float(r["mean_across_severity"])
            for r in per_seed_family
            if r["family"] == family and r["metric"] == metric
        ]
        aggregate.append({
            "family": family,
            "metric": metric,
            "n_seeds": len(vals),
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "positive_seeds": sum(v > 0 for v in vals),
        })
    write_csv(outdir / "03_three_seed_family_summary.csv", aggregate)

    # Condition-wise 3-seed mean/SD.
    condition_summary = []
    for condition in [condition_name(k, s) for k, s in conditions]:
        rr = [r for r in all_rows if r["condition"] == condition]
        for metric in [
            "source_top1", "final_top1", "delta_top1_pp",
            "joint_retention", "ento_to_ento_neural",
            "mvl_to_mvl_neural", "joint_to_clean_behavior",
        ]:
            vals = [float(x[metric]) for x in rr]
            condition_summary.append({
                "condition": condition,
                "metric": metric,
                "n_seeds": len(vals),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "positive_seeds": sum(v > 0 for v in vals),
            })
    write_csv(outdir / "04_three_seed_condition_summary.csv", condition_summary)

    protocol = {
        "architecture": (
            "MVL-like output + calibrated ENTO-like residual across the complete class space; "
            "interaction OFF; extra DeltaM OFF"
        ),
        "evaluation_only": True,
        "robustness_used_for_training_or_checkpoint_selection": False,
        "canonical_inference": "historical_amp",
        "corruption_seed": a.corruption_seed,
        "profile": a.profile,
        "conditions": [
            {"condition": condition_name(k, s), "family": k, "severity": s}
            for k, s in conditions
        ],
        "fixed_current_corruption_implementation": {
            "occlusion": (
                "one deterministic square; severity is occluded AREA fraction; "
                "fill RGB=[123,116,103]"
            ),
            "patch_shuffle": (
                "4x4 grid; severity is fraction of patch positions participating "
                "in a deterministic within-image permutation"
            ),
            "gaussian_blur": "PIL GaussianBlur radius equals severity",
            "gaussian_noise": "RGB [0,1] additive Gaussian sigma equals severity",
            "color_shift": (
                "HSV hue rotation; shift = severity*0.08 of a full hue cycle "
                "(severity .45 ≈ 13 degrees)"
            ),
            "rng": (
                "per-image SHA256-derived seed from corruption_seed, family, "
                "severity and image identity; identical perturbations across model seeds"
            ),
        },
        "important_scope_note": (
            "The five family names and severity labels match the prior analysis convention, "
            "but this package documents and locks the CURRENT implementation. Do not call it "
            "bitwise-identical to an older robustness script unless that older source is audited."
        ),
        "representation_rdm": "Pearson correlation distance (1-r)",
        "representation_retention": (
            "Spearman(clean model RDM, perturbed model RDM)"
        ),
        "neural_alignment": (
            "Spearman(perturbed ENTO RDM, ENTO neural RDM) and "
            "Spearman(perturbed MVL RDM, MVL neural RDM)"
        ),
        "behavior_under_perturbation_note": (
            "Corrupted model RDM versus CLEAN behavior RDM is reported only as "
            "preservation of behaviorally relevant geometry; it is NOT a prediction "
            "of corrupted-condition pigeon behavior."
        ),
        "seeds": seeds,
        "max_test_images": a.max_test_images,
        "official_eval_dir": str(eval_dir),
        "manifest": str(manifest_path),
    }
    write_json(outdir / "PROTOCOL.json", protocol)

    print("\n[ALL DONE]")
    print("Per-condition:", outdir / "01_per_seed_condition_results.csv")
    print("Family means :", outdir / "02_per_seed_family_means.csv")
    print("3-seed family:", outdir / "03_three_seed_family_summary.csv")
    print("3-seed cond. :", outdir / "04_three_seed_condition_summary.csv")
    print("Protocol     :", outdir / "PROTOCOL.json")


if __name__ == "__main__":
    main()
