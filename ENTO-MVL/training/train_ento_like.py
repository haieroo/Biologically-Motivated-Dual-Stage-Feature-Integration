# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.resolve().parents[1]
CORE = REPO_ROOT / "core"
sys.path.insert(0, str(CORE))

from seed_utils import resolve_seed
from cub_attribute_dataset import build_attribute_datasets
from models_single_region_branch import IndependentAutoRegionBranchResNet50


def parse_args():
    p = argparse.ArgumentParser(
        "Formal independent ENTO-like 448 training: full train, fixed epoch 12."
    )
    p.add_argument("--data-root", required=True)
    p.add_argument("--cub-root", required=True)
    p.add_argument("--clean-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=None, help="Optional; OS-random if omitted.")

    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--num-regions", type=int, default=4)
    p.add_argument("--region-dim", type=int, default=256)
    p.add_argument("--branch-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.10)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--aux-weight", type=float, default=0.20)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=0.5)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    s = torch.initial_seed() % (2**32)
    random.seed(s)
    np.random.seed(s)


def make_loader(ds, batch, workers, seed):
    g = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=int(batch),
        shuffle=True,
        drop_last=True,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(workers) > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )


def amp_ctx(enabled: bool):
    if torch.cuda.is_available():
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=enabled)
    from contextlib import nullcontext
    return nullcontext()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def append_csv(path: Path, row):
    first = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def main():
    a = parse_args()
    a.seed = resolve_seed(a.seed, "ENTO-like")
    out = Path(a.output_dir).resolve()
    if out.exists() and a.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(a.seed)

    # Match the historical independent branch training reproducibility path.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    train_set, official_eval_set, attribute_names = build_attribute_datasets(
        data_root=a.data_root,
        cub_root=a.cub_root,
        img_size=a.img_size,
        random_erasing=False,
        attribute_mode="basic",
    )
    if len(train_set) != 5994:
        raise RuntimeError(f"Expected 5994 training images, got {len(train_set)}")
    if len(official_eval_set) != 5794:
        raise RuntimeError(
            f"Expected 5794 official evaluation images, got {len(official_eval_set)}"
        )

    # IMPORTANT: official_eval_set is intentionally not used below.
    loader = make_loader(
        train_set, a.batch_size, a.num_workers, a.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IndependentAutoRegionBranchResNet50(
        num_classes=len(train_set.classes),
        region="ento",
        num_regions=a.num_regions,
        region_dim=a.region_dim,
        branch_dim=a.branch_dim,
        dropout=a.dropout,
    )
    load_result = model.load_clean(a.clean_checkpoint, prefer_ema=False)
    model.freeze_backbone()

    trainable_names = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    forbidden = [
        name for name in trainable_names
        if not (
            name.startswith("region_encoder.")
            or name.startswith("branch_classifier.")
            or name == "alpha"
        )
    ]
    if forbidden:
        raise RuntimeError(f"Unexpected trainable parameters: {forbidden[:20]}")
    if float(model.alpha.detach()) != 0.0:
        raise RuntimeError("ENTO alpha must initialize exactly at zero.")

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=a.lr,
        weight_decay=a.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(a.epochs, 1)
    )
    scaler = make_scaler(a.amp and device.type == "cuda")

    print("=" * 140)
    print("FORMAL INDEPENDENT ENTO-LIKE | 3-SEED ABLATION PROTOCOL")
    print(f"seed={a.seed} | 448x448 | full train=5994 | fixed epoch={a.epochs}")
    print("Initialization: same Clean ResNet50 checkpoint")
    print("Backbone: fully frozen")
    print("ENTO: layer2 automatic-region branch, independently class-supervised")
    print("Formula: E = z_base + alpha_E * z_E(logits)")
    print("Neural RDM loss: OFF")
    print("Attribute/part/bbox supervision: OFF")
    print("Official 5794: NOT used in training/checkpoint selection")
    print(
        f"batch={a.batch_size} | accum={a.accum_steps} | "
        f"lr={a.lr:g} | wd={a.weight_decay:g} | aux={a.aux_weight:g}"
    )
    print(
        f"load_clean missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)}"
    )
    print(f"trainable params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 140)

    for epoch in range(1, a.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        n_total = 0
        loss_sum = 0.0
        final_sum = 0.0
        branch_sum = 0.0

        for step, batch in enumerate(loader, 1):
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp_ctx(a.amp and device.type == "cuda"):
                pred = model(images)
                final_ce = F.cross_entropy(
                    pred["logits"],
                    labels,
                    label_smoothing=a.label_smoothing,
                )
                branch_ce = F.cross_entropy(
                    pred["logits_branch"],
                    labels,
                    label_smoothing=a.label_smoothing,
                )
                loss = final_ce + a.aux_weight * branch_ce

            scaler.scale(loss / a.accum_steps).backward()

            if step % a.accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), a.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            n = int(labels.numel())
            n_total += n
            loss_sum += float(loss.detach()) * n
            final_sum += float(final_ce.detach()) * n
            branch_sum += float(branch_ce.detach()) * n

        scheduler.step()
        row = {
            "epoch": epoch,
            "train_total": loss_sum / max(n_total, 1),
            "train_final_ce": final_sum / max(n_total, 1),
            "train_branch_ce": branch_sum / max(n_total, 1),
            "alpha_E": float(model.alpha.detach().cpu()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_images_effective": n_total,
        }
        append_csv(out / "train_log.csv", row)
        print(
            f"ENTO | {epoch:03d}/{a.epochs} | "
            f"loss={row['train_total']:.5f} | "
            f"finalCE={row['train_final_ce']:.5f} | "
            f"branchCE={row['train_branch_ce']:.5f} | "
            f"alpha_E={row['alpha_E']:+.5f} | "
            f"lr={row['lr']:.3e}"
        )

    payload = {
        "model": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "epoch": a.epochs,
        "seed": a.seed,
        "region": "ento",
        "classes": list(train_set.classes),
        "attribute_names": list(attribute_names or []),
        "architecture": "IndependentAutoRegionBranchResNet50",
        "formula": "logits = logits_base + alpha_E * logits_ento_branch",
        "rdm_weight": 0.0,
        "attribute_supervision_used": False,
        "part_supervision_used": False,
        "bbox_supervision_used": False,
        "official_test_used_for_training": False,
        "official_test_used_for_checkpoint_selection": False,
        "checkpoint_selection": f"fixed final epoch {a.epochs}",
        "training_images": len(train_set),
        "args": vars(a),
    }
    torch.save(payload, out / "final_epoch12.pth")

    summary = {
        "experiment": "formal_independent_ento_like_fixed12",
        "seed": a.seed,
        "checkpoint": str(out / "final_epoch12.pth"),
        "formula": payload["formula"],
        "training_images": len(train_set),
        "official_eval_images": len(official_eval_set),
        "official_test_used_for_training_or_selection": False,
        "fixed_epoch": a.epochs,
        "alpha_E_final": float(model.alpha.detach().cpu()),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "trainable_names": trainable_names,
        "clean_checkpoint": str(Path(a.clean_checkpoint).resolve()),
        "protocol_note": (
            "Current formal ENTO-like ablation: full 5994 training set, fixed epoch 12. "
            "This is intentionally not the older 50/50 calibration-selected ENTO run."
        ),
    }
    (out / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[TRAIN DONE]")
    print(out / "final_epoch12.pth")
    print("Official 5794 has not been evaluated by this training script.")


if __name__ == "__main__":
    main()
