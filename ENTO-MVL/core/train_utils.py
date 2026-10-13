from __future__ import annotations

import copy
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.total / max(self.count, 1)

    def update(self, value, n=1):
        self.total += float(value) * int(n)
        self.count += int(n)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, labels, topk=(1, 5)):
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1))
    results = []
    for k in topk:
        k = min(k, logits.size(1))
        results.append(
            correct[:k].reshape(-1).float().sum() * (100.0 / labels.size(0))
        )
    return results


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def freeze_batch_norm_stats(model: nn.Module):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            module.eval()


def one_hot(labels, num_classes, smoothing=0.0):
    smoothing = float(max(0.0, min(0.99, smoothing)))
    off = smoothing / num_classes
    on = 1.0 - smoothing + off
    target = torch.full(
        (labels.size(0), num_classes),
        off,
        device=labels.device,
        dtype=torch.float32,
    )
    target.scatter_(1, labels[:, None], on)
    return target


def soft_target_cross_entropy(logits, targets):
    return -(
        targets * F.log_softmax(logits.float(), dim=1)
    ).sum(dim=1).mean()


def kd_kl_loss(student_logits, teacher_logits, temperature=3.0):
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits.float() / t, dim=1),
        F.softmax(teacher_logits.detach().float() / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def rand_bbox(size, lam):
    _, _, h, w = size
    ratio = math.sqrt(1.0 - lam)
    cut_w = int(w * ratio)
    cut_h = int(h * ratio)
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, h)
    return x1, y1, x2, y2


def apply_mixup_cutmix(
    images,
    labels,
    num_classes,
    mix_prob=0.4,
    mixup_alpha=0.2,
    cutmix_alpha=0.8,
    cutmix_switch_prob=0.5,
    label_smoothing=0.1,
):
    targets = one_hot(labels, num_classes, label_smoothing)
    if images.size(0) < 2 or random.random() > mix_prob:
        return images, targets

    perm = torch.randperm(images.size(0), device=images.device)
    use_cutmix = random.random() < cutmix_switch_prob

    if use_cutmix and cutmix_alpha > 0:
        lam = torch.distributions.Beta(
            cutmix_alpha, cutmix_alpha
        ).sample().item()
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        area = max((x2 - x1) * (y2 - y1), 0)
        lam = 1.0 - area / float(images.size(2) * images.size(3))
        targets = targets * lam + targets[perm] * (1.0 - lam)
        return mixed, targets

    lam = torch.distributions.Beta(
        mixup_alpha, mixup_alpha
    ).sample().item()
    mixed = images * lam + images[perm] * (1.0 - lam)
    targets = targets * lam + targets[perm] * (1.0 - lam)
    return mixed, targets


def autocast_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(
                "cuda", enabled=enabled, init_scale=512.0
            )
        except TypeError:
            return torch.amp.GradScaler(
                enabled=enabled, init_scale=512.0
            )
    return torch.cuda.amp.GradScaler(
        enabled=enabled, init_scale=512.0
    )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        current = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            value = current[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.copy_(
                    ema_value * self.decay
                    + value * (1.0 - self.decay)
                )
            else:
                ema_value.copy_(value)


def build_warmup_cosine_scheduler(
    optimizer,
    total_epochs: int,
    warmup_epochs: int,
):
    def factor(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (
            float(epoch - warmup_epochs)
            / float(max(1, total_epochs - warmup_epochs))
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=factor,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    amp: bool,
):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(amp and device.type == "cuda"):
            output = model(images)
            logits = output["logits"]
            loss = F.cross_entropy(logits, labels)
        a1, a5 = accuracy(logits, labels, (1, 5))
        n = labels.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(a1.item(), n)
        top5_meter.update(a5.item(), n)

    return {
        "val_loss": loss_meter.avg,
        "val_top1": top1_meter.avg,
        "val_top5": top5_meter.avg,
    }


def save_training_checkpoint(
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
        "epoch": epoch,
        "model": (
            ema.module.state_dict()
            if ema is not None
            else model.state_dict()
        ),
        "raw_model": model.state_dict(),
        "ema_model": (
            ema.module.state_dict()
            if ema is not None else None
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_top1": float(best_top1),
        "args": vars(args),
        "classes": classes,
        "best_is_ema": ema is not None,
    }
    torch.save(state, path)
