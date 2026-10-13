from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path
import sys

REPO_ROOT=Path(__file__).resolve().parents[1]
CORE=REPO_ROOT/"core"
if str(CORE) not in sys.path:
    sys.path.insert(0,str(CORE))

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seed_utils import resolve_seed
from data_pipeline import build_datasets
from models_compact_mvl import (
    CompactMVLResNet50,
    VALID_MVL_MODES,
    load_clean_checkpoint,
    mode_components,
)
from train_utils import (
    AverageMeter,
    ModelEMA,
    append_csv,
    apply_mixup_cutmix,
    autocast_context,
    build_warmup_cosine_scheduler,
    freeze_batch_norm_stats,
    make_scaler,
    save_json,
    set_seed,
    soft_target_cross_entropy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Matched compact MVL selection and formal ablation at 448."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clean_checkpoint", required=True)
    parser.add_argument(
        "--ablation_mode",
        choices=sorted(VALID_MVL_MODES),
        required=True,
    )
    parser.add_argument("--img_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--plugin_warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--plugin_lr", type=float, default=2e-4)
    parser.add_argument("--head_lr", type=float, default=2e-5)
    parser.add_argument("--layer4_lr", type=float, default=8e-6)
    parser.add_argument("--layer3_lr", type=float, default=4e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--aux_weight", type=float, default=0.30)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--mix_prob", type=float, default=0.25)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None, help="Optional; OS-random if omitted.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    seed: int,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def set_phase_trainability(
    model,
    plugin_only: bool,
):
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.mvl_plugin.parameters():
        parameter.requires_grad_(True)
    model.alpha_mvl.requires_grad_(True)

    if not plugin_only:
        for parameter in model.layer4.parameters():
            parameter.requires_grad_(True)
        for parameter in model.layer3.parameters():
            parameter.requires_grad_(True)
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)


def build_optimizer(
    model,
    args,
    plugin_only: bool,
):
    groups = [
        {
            "params": list(model.mvl_plugin.parameters())
            + [model.alpha_mvl],
            "lr": args.plugin_lr,
            "name": "mvl_plugin",
        }
    ]
    if not plugin_only:
        groups.extend(
            [
                {
                    "params": model.classifier.parameters(),
                    "lr": args.head_lr,
                    "name": "classifier",
                },
                {
                    "params": model.layer4.parameters(),
                    "lr": args.layer4_lr,
                    "name": "layer4",
                },
                {
                    "params": model.layer3.parameters(),
                    "lr": args.layer3_lr,
                    "name": "layer3",
                },
            ]
        )
    return torch.optim.AdamW(
        groups,
        weight_decay=args.weight_decay,
    )


@torch.no_grad()
def evaluate_argmax(
    model,
    loader,
    device,
    amp: bool,
    return_predictions: bool = False,
):
    model.eval()
    loss_meter = AverageMeter()
    labels_all: List[torch.Tensor] = []
    predictions_all: List[torch.Tensor] = []
    logits_top5_all: List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(amp and device.type == "cuda"):
            output = model(images)
            logits = output["logits"]
            loss = F.cross_entropy(
                logits.float(),
                labels,
            )
        prediction = logits.argmax(dim=1)
        top5 = logits.topk(
            min(5, logits.size(1)),
            dim=1,
        ).indices

        loss_meter.update(loss.item(), labels.size(0))
        labels_all.append(labels.cpu())
        predictions_all.append(prediction.cpu())
        logits_top5_all.append(top5.cpu())

    labels_tensor = torch.cat(labels_all)
    predictions_tensor = torch.cat(predictions_all)
    top5_tensor = torch.cat(logits_top5_all)

    correct = predictions_tensor.eq(labels_tensor)
    top5_correct = top5_tensor.eq(
        labels_tensor[:, None]
    ).any(dim=1)

    result = {
        "val_loss": loss_meter.avg,
        "val_top1": float(
            correct.float().mean().item() * 100.0
        ),
        "val_top5": float(
            top5_correct.float().mean().item() * 100.0
        ),
        "correct_count": int(correct.sum().item()),
        "sample_count": int(labels_tensor.numel()),
    }
    if return_predictions:
        result.update(
            {
                "labels": labels_tensor,
                "predictions": predictions_tensor,
                "correct": correct,
            }
        )
    return result


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    ema,
    optimizer,
    scheduler,
    best_top1: float,
    args,
    classes,
):
    state = {
        "epoch": int(epoch),
        "model": ema.module.state_dict(),
        "raw_model": model.state_dict(),
        "ema_model": ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_top1": float(best_top1),
        "args": vars(args),
        "classes": classes,
        "best_is_ema": True,
        "ablation_mode": args.ablation_mode,
    }
    torch.save(state, path)


def exact_mcnemar_p(corrected: int, degraded: int) -> float:
    discordant = int(corrected + degraded)
    if discordant == 0:
        return 1.0
    tail = min(int(corrected), int(degraded))
    probability = sum(
        math.comb(discordant, k)
        for k in range(tail + 1)
    ) / float(2 ** discordant)
    return min(1.0, 2.0 * probability)


def write_predictions(
    path: Path,
    source_result: Dict,
    final_result: Dict,
):
    rows = []
    labels = final_result["labels"]
    source_predictions = source_result["predictions"]
    final_predictions = final_result["predictions"]
    source_correct = source_result["correct"]
    final_correct = final_result["correct"]

    for index in range(labels.numel()):
        rows.append(
            {
                "image_index": index,
                "true_label": int(labels[index].item()),
                "source_prediction": int(
                    source_predictions[index].item()
                ),
                "final_prediction": int(
                    final_predictions[index].item()
                ),
                "source_correct": int(
                    source_correct[index].item()
                ),
                "final_correct": int(
                    final_correct[index].item()
                ),
            }
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.seed = resolve_seed(args.seed, "MVL-like")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = build_datasets(
        args.data_root,
        img_size=args.img_size,
        random_erasing=True,
    )
    classes = train_set.classes
    num_classes = len(classes)

    train_loader = build_loader(
        train_set,
        args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = build_loader(
        val_set,
        args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        seed=args.seed + 10000,
    )

    save_json(output_dir / "args.json", vars(args))
    save_json(output_dir / "classes.json", classes)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = CompactMVLResNet50(
        num_classes=num_classes,
        mode=args.ablation_mode,
        pretrained=False,
    ).to(device)

    load_result = load_clean_checkpoint(
        model,
        args.clean_checkpoint,
        prefer_ema=True,
    )

    initial = evaluate_argmax(
        model,
        val_loader,
        device,
        args.amp,
        return_predictions=True,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    plugin_parameters = sum(
        parameter.numel()
        for parameter in model.mvl_plugin.parameters()
    )

    print("=" * 140)
    print(
        f"FORMAL MVL STRUCTURAL ABLATION | "
        f"mode={args.ablation_mode} | seed={args.seed}"
    )
    print(
        "Teacher=0 | KD=0 | ENTO=0 | "
        "all groups initialized from the same Clean ResNet50"
    )
    print(
        f"Clean load missing={len(load_result.missing_keys)} | "
        f"unexpected={len(load_result.unexpected_keys)}"
    )
    print(
        f"Protected clean equivalence | "
        f"Top-1={initial['val_top1']:.4f} | "
        f"Top-5={initial['val_top5']:.4f} | "
        f"Correct={initial['correct_count']}/{initial['sample_count']} | "
        f"alpha_MVL={model.alpha_mvl.item():.6f}"
    )
    print(
        f"Total params={total_parameters} | "
        f"Declared MVL params={plugin_parameters}"
    )
    print("=" * 140)

    plugin_only = args.plugin_warmup_epochs > 0
    set_phase_trainability(
        model,
        plugin_only=plugin_only,
    )
    optimizer = build_optimizer(
        model,
        args,
        plugin_only=plugin_only,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    scaler = make_scaler(
        args.amp and device.type == "cuda"
    )
    ema = ModelEMA(
        model,
        decay=args.ema_decay,
    )

    best_top1 = initial["val_top1"]
    best_top5 = initial["val_top5"]
    best_epoch = 0
    save_checkpoint(
        output_dir / "best.pth",
        0,
        model,
        ema,
        optimizer,
        scheduler,
        best_top1,
        args,
        classes,
    )

    log_path = output_dir / "train_log.csv"

    for epoch in range(1, args.epochs + 1):
        if (
            plugin_only
            and epoch == args.plugin_warmup_epochs + 1
        ):
            plugin_only = False
            set_phase_trainability(
                model,
                plugin_only=False,
            )
            optimizer = build_optimizer(
                model,
                args,
                plugin_only=False,
            )
            scheduler = build_warmup_cosine_scheduler(
                optimizer,
                total_epochs=max(
                    1,
                    args.epochs
                    - args.plugin_warmup_epochs,
                ),
                warmup_epochs=1,
            )
            print(
                "Unfroze layer3/layer4/classifier."
            )

        model.train()
        freeze_batch_norm_stats(model)
        loss_meter = AverageMeter()
        final_ce_meter = AverageMeter()
        auxiliary_meter = AverageMeter()
        train_correct = 0
        train_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )
            images, soft_targets = apply_mixup_cutmix(
                images,
                labels,
                num_classes,
                mix_prob=args.mix_prob,
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                label_smoothing=args.label_smoothing,
            )

            with autocast_context(
                args.amp and device.type == "cuda"
            ):
                output = model(images)
                final_ce = soft_target_cross_entropy(
                    output["logits"],
                    soft_targets,
                )
                auxiliary = soft_target_cross_entropy(
                    output["logits_mvl"],
                    soft_targets,
                )
                loss = (
                    final_ce
                    + args.aux_weight * auxiliary
                )
                scaled_loss = loss / args.accum_steps

            scaler.scale(scaled_loss).backward()

            should_step = (
                step % args.accum_steps == 0
                or step == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                trainable = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                gradient_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        args.grad_clip,
                    )
                )
                if torch.isfinite(gradient_norm):
                    scaler.step(optimizer)
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

            prediction = output["logits"].detach().argmax(dim=1)
            train_correct += int(
                prediction.eq(labels).sum().item()
            )
            train_count += labels.numel()
            loss_meter.update(
                loss.item(),
                labels.size(0),
            )
            final_ce_meter.update(
                final_ce.item(),
                labels.size(0),
            )
            auxiliary_meter.update(
                auxiliary.item(),
                labels.size(0),
            )

        raw_validation = evaluate_argmax(
            model,
            val_loader,
            device,
            args.amp,
        )
        ema_validation = evaluate_argmax(
            ema.module,
            val_loader,
            device,
            args.amp,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "ablation_mode": args.ablation_mode,
            "seed": args.seed,
            "plugin_only": int(plugin_only),
            "train_loss": loss_meter.avg,
            "train_final_ce": final_ce_meter.avg,
            "train_auxiliary": auxiliary_meter.avg,
            "train_top1_hard": (
                train_correct / max(train_count, 1) * 100.0
            ),
            "raw_val_top1": raw_validation["val_top1"],
            "ema_val_top1": ema_validation["val_top1"],
            "ema_val_top5": ema_validation["val_top5"],
            "ema_correct_count": ema_validation["correct_count"],
            "alpha_mvl": float(
                ema.module.alpha_mvl.item()
            ),
            "plugin_lr": optimizer.param_groups[0]["lr"],
        }
        append_csv(log_path, row)

        print(
            f"MVL-ABL | {epoch:03d}/{args.epochs} | "
            f"Mode={args.ablation_mode} | "
            f"Top-1={row['ema_val_top1']:.4f} | "
            f"Gain={row['ema_val_top1'] - initial['val_top1']:+.4f} | "
            f"Top-5={row['ema_val_top5']:.4f} | "
            f"Correct={row['ema_correct_count']} | "
            f"alpha={row['alpha_mvl']:.4f}"
        )

        save_checkpoint(
            output_dir / "last.pth",
            epoch,
            model,
            ema,
            optimizer,
            scheduler,
            best_top1,
            args,
            classes,
        )

        improved = (
            ema_validation["val_top1"] > best_top1 + 1e-12
            or (
                abs(
                    ema_validation["val_top1"]
                    - best_top1
                ) <= 1e-12
                and ema_validation["val_top5"] > best_top5
            )
        )
        if improved:
            best_top1 = ema_validation["val_top1"]
            best_top5 = ema_validation["val_top5"]
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pth",
                epoch,
                model,
                ema,
                optimizer,
                scheduler,
                best_top1,
                args,
                classes,
            )
            print("  Saved new best.")

    best_checkpoint = torch.load(
        output_dir / "best.pth",
        map_location="cpu",
    )
    model.load_state_dict(
        best_checkpoint["model"],
        strict=True,
    )
    model.to(device)

    final = evaluate_argmax(
        model,
        val_loader,
        device,
        args.amp,
        return_predictions=True,
    )

    source_correct = initial["correct"]
    final_correct = final["correct"]
    corrected = int(
        ((~source_correct) & final_correct).sum().item()
    )
    degraded = int(
        (source_correct & (~final_correct)).sum().item()
    )
    net = corrected - degraded
    count_difference = (
        final["correct_count"]
        - initial["correct_count"]
    )
    if count_difference != net:
        raise RuntimeError(
            "Argmax accounting mismatch: "
            f"count difference={count_difference}, net={net}"
        )

    write_predictions(
        output_dir / "best_val_predictions.csv",
        initial,
        final,
    )

    summary = {
        "experiment": "compact_mvl_finalization",
        "ablation_mode": args.ablation_mode,
        "seed": args.seed,
        "clean_top1": initial["val_top1"],
        "clean_top5": initial["val_top5"],
        "clean_correct_count": initial["correct_count"],
        "best_top1": final["val_top1"],
        "best_top5": final["val_top5"],
        "best_correct_count": final["correct_count"],
        "gain_vs_clean": (
            final["val_top1"]
            - initial["val_top1"]
        ),
        "best_epoch": int(best_checkpoint["epoch"]),
        "corrected_count": corrected,
        "degraded_count": degraded,
        "net_corrected": net,
        "mcnemar_exact_p_vs_clean": exact_mcnemar_p(
            corrected,
            degraded,
        ),
        "total_parameters": total_parameters,
        "declared_mvl_parameters": plugin_parameters,
        "teacher_used": False,
        "kd_used": False,
        "ento_used": False,
        "same_clean_initialization": True,
        "top1_rule": "argmax",
        "components": mode_components(args.ablation_mode),
    }
    save_json(
        output_dir / "summary.json",
        summary,
    )

    print("=" * 140)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 140)


if __name__ == "__main__":
    main()
