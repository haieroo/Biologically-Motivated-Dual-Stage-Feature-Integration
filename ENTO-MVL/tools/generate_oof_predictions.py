from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import random
import shutil
from pathlib import Path
import sys

REPO_ROOT=Path(__file__).resolve().parents[1]
CORE=REPO_ROOT/"core"
if str(CORE) not in sys.path:
    sys.path.insert(0,str(CORE))


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from seed_utils import resolve_seed
from cub_attribute_dataset import build_attribute_datasets
from selector_transform import install_selector_preserving_transform
from data_pipeline import build_transforms
from models_compact_mvl import CompactMVLResNet50


def parse_args():
    parser = argparse.ArgumentParser(
        "Generate leakage-free 5-fold OOF Compact-MVL predictions."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cub_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--attribute_mode", default="basic")
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--split_seed", type=int, default=1731)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def autocast_context(enabled):
    if torch.cuda.is_available():
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=enabled)
    from contextlib import nullcontext
    return nullcontext()


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def dataset_labels(dataset):
    imagefolder = getattr(dataset, "imagefolder", None)
    if imagefolder is not None and hasattr(imagefolder, "targets"):
        return list(imagefolder.targets)
    return [int(dataset[index][1]) for index in range(len(dataset))]


def stratified_folds(labels, folds, split_seed):
    labels = torch.as_tensor(labels, dtype=torch.long)
    fold_indices = [[] for _ in range(folds)]
    for class_index in labels.unique(sorted=True).tolist():
        indices = torch.where(labels == int(class_index))[0]
        generator = torch.Generator()
        generator.manual_seed(int(split_seed) + int(class_index) * 1009)
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        for offset, index in enumerate(shuffled.tolist()):
            fold_indices[offset % folds].append(index)
    return [sorted(values) for values in fold_indices]


def build_loader(dataset, batch_size, shuffle, drop_last, workers, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def forward_logits(model, images):
    features = model.forward_features(images)
    f3 = features["f3"]
    f4 = features["f4"]
    z_base = model.pool(f4).flatten(1)
    logits_base = model.classifier(z_base)
    _, logits_mvl, _ = model.mvl_plugin(f3, f4)
    return logits_base + model.alpha_mvl * logits_mvl


def build_optimizer(model, args):
    head_names = ("classifier.", "mvl_plugin.", "alpha_mvl")
    backbone = []
    heads = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(head_names) or name == "alpha_mvl":
            heads.append(parameter)
        else:
            backbone.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.backbone_lr, "name": "backbone"},
            {"params": heads, "lr": args.head_lr, "name": "heads"},
        ],
        weight_decay=args.weight_decay,
    )


def train_fold(model, loader, optimizer, scaler, scheduler, device, args, fold):
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        correct = 0
        count = 0
        for step, batch in enumerate(loader, start=1):
            images, labels, _, _, _ = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(args.amp and device.type == "cuda"):
                logits = forward_logits(model, images)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=args.label_smoothing,
                )
            scaler.scale(loss / args.accum_steps).backward()
            if step % args.accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            n = labels.size(0)
            loss_sum += float(loss.detach().item()) * n
            correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            count += n
        scheduler.step()
        print(
            f"OOF fold={fold + 1}/{args.folds} | "
            f"epoch={epoch:02d}/{args.epochs} | "
            f"loss={loss_sum / max(count, 1):.4f} | "
            f"train_top1={100.0 * correct / max(count, 1):.2f}"
        )


@torch.no_grad()
def predict_fold(model, loader, device, amp):
    model.eval()
    rows = []
    for batch in loader:
        images, labels, _, _, image_ids = batch
        images = images.to(device, non_blocking=True)
        with autocast_context(amp and device.type == "cuda"):
            logits = forward_logits(model, images)
        probabilities = F.softmax(logits.float(), dim=1)
        top5_probability, top5_index = probabilities.topk(5, dim=1)
        predictions = top5_index[:, 0]
        for i in range(labels.size(0)):
            rows.append(
                {
                    "image_id": int(image_ids[i].item()),
                    "label": int(labels[i].item()),
                    "source_prediction": int(predictions[i].cpu().item()),
                    "source_top5": "|".join(
                        str(int(value))
                        for value in top5_index[i].cpu().tolist()
                    ),
                    "source_top5_probability": "|".join(
                        f"{float(value):.8f}"
                        for value in top5_probability[i].cpu().tolist()
                    ),
                }
            )
    return rows


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.seed = resolve_seed(args.seed, "OOF")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    output_dir = Path(args.output_dir)
    output_csv = Path(args.output_csv)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    train_set, _, _ = build_attribute_datasets(
        data_root=args.data_root,
        cub_root=args.cub_root,
        img_size=args.img_size,
        random_erasing=False,
        attribute_mode=args.attribute_mode,
    )
    # Separate deterministic view of the same official training images for
    # leakage-free fold prediction. OOF predictions must not use random crop,
    # flip, or other training-time augmentation.
    train_eval_set, _, _ = build_attribute_datasets(
        data_root=args.data_root,
        cub_root=args.cub_root,
        img_size=args.img_size,
        random_erasing=False,
        attribute_mode=args.attribute_mode,
    )
    install_selector_preserving_transform(train_set, args.img_size)
    _, validation_transform = build_transforms(
        img_size=args.img_size,
        random_erasing=False,
    )
    train_eval_set.imagefolder.transform = validation_transform
    labels = dataset_labels(train_set)
    if dataset_labels(train_eval_set) != labels:
        raise RuntimeError("Train/eval OOF dataset label order differs.")
    folds = stratified_folds(labels, args.folds, args.split_seed)
    all_indices = set(range(len(train_set)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows = []
    for fold, heldout_indices in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = fold_dir / "oof_predictions.csv"
        if prediction_path.exists():
            with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
                all_rows.extend(list(csv.DictReader(handle)))
            print(f"Skip completed OOF fold {fold + 1}: {prediction_path}")
            continue

        fold_seed = int(args.seed) + fold * 10007
        set_seed(fold_seed)
        train_indices = sorted(all_indices.difference(heldout_indices))
        train_loader = build_loader(
            Subset(train_set, train_indices),
            args.batch_size,
            True,
            True,
            args.num_workers,
            fold_seed,
        )
        heldout_loader = build_loader(
            Subset(train_eval_set, heldout_indices),
            max(args.batch_size, 2),
            False,
            False,
            args.num_workers,
            fold_seed + 1000,
        )

        model = CompactMVLResNet50(
            num_classes=len(train_set.classes),
            mode="compact_mvl",
            pretrained=True,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
        ).to(device)
        optimizer = build_optimizer(model, args)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
        )
        scaler = make_scaler(args.amp and device.type == "cuda")

        train_fold(
            model,
            train_loader,
            optimizer,
            scaler,
            scheduler,
            device,
            args,
            fold,
        )
        rows = predict_fold(
            model,
            heldout_loader,
            device,
            args.amp,
        )
        for row in rows:
            row["fold"] = fold + 1
            row["seed"] = args.seed
        rows.sort(key=lambda row: int(row["image_id"]))
        write_csv(prediction_path, rows)
        all_rows.extend(rows)
        del model, optimizer, scheduler, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Normalize values loaded from previous fold CSVs and verify one OOF row
    # per official training image.
    normalized = []
    seen = set()
    for row in all_rows:
        image_id = int(row["image_id"])
        if image_id in seen:
            raise RuntimeError(f"Duplicate OOF image_id: {image_id}")
        seen.add(image_id)
        normalized.append(
            {
                "image_id": image_id,
                "label": int(row["label"]),
                "source_prediction": int(row["source_prediction"]),
                "source_top5": row["source_top5"],
                "source_top5_probability": row["source_top5_probability"],
                "fold": int(row["fold"]),
                "seed": int(row["seed"]),
            }
        )
    if len(normalized) != len(train_set):
        raise RuntimeError(
            f"OOF row count {len(normalized)} != train set {len(train_set)}"
        )
    normalized.sort(key=lambda row: row["image_id"])
    write_csv(output_csv, normalized)

    correct = sum(
        row["source_prediction"] == row["label"]
        for row in normalized
    )
    summary = {
        "seed": args.seed,
        "folds": args.folds,
        "epochs_per_fold": args.epochs,
        "training_images": len(train_set),
        "oof_top1": 100.0 * correct / len(normalized),
        "leakage_free": True,
        "full_source_checkpoint_used": False,
        "fixed_epoch_training": True,
        "output_csv": str(output_csv),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
