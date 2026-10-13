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
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]
CORE = REPO_ROOT / "core"
sys.path.insert(0, str(CORE))

from cub_attribute_dataset import build_attribute_datasets
from data_pipeline import build_transforms
from models_ento_mvl import ENTOMVL
from neural_rdm_utils import load_confuser_map, choose_oof_confuser
from noattr_modules import region_overlap_loss
from seed_utils import resolve_seed


def parse_args():
    p = argparse.ArgumentParser(
        "Final all-class ENTO-MVL training: frozen MVL-like source + ENTO residual adaptation"
    )
    p.add_argument("--data-root", required=True)
    p.add_argument("--cub-root", required=True)
    p.add_argument("--source-checkpoint", required=True)
    p.add_argument("--confuser-map", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--seed", type=int, default=None,
        help="Optional; if omitted, infer the matched seed from the source checkpoint.",
    )

    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--num-regions", type=int, default=4)
    p.add_argument("--region-dim", type=int, default=256)
    p.add_argument("--branch-dim", type=int, default=512)
    p.add_argument("--residual-hidden-dim", type=int, default=256)
    p.add_argument("--residual-dropout", type=float, default=0.10)
    p.add_argument("--residual-logit-clip", type=float, default=1.0)
    p.add_argument("--alpha-e", type=float, default=0.20)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--ento-aux-weight", type=float, default=0.20)

    # Auxiliary objectives retained from the reported all-class training run.
    p.add_argument("--confuser-weight", type=float, default=0.10)
    p.add_argument("--confuser-margin", type=float, default=0.20)
    p.add_argument("--preserve-weight", type=float, default=0.30)
    p.add_argument("--repair-weight", type=float, default=0.10)
    p.add_argument("--repair-margin", type=float, default=0.10)
    p.add_argument("--region-overlap-weight", type=float, default=0.01)
    p.add_argument("--residual-l2-weight", type=float, default=1e-3)

    # Training-only, label-free RMS calibration split.
    p.add_argument("--scale-calibration-fraction", type=float, default=0.10)
    p.add_argument("--scale-split-seed", type=int, default=5204096)

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


def autocast_context(enabled: bool):
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


def make_loader(ds, batch, shuffle, drop_last, workers, seed):
    g = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=int(batch),
        shuffle=bool(shuffle),
        drop_last=bool(drop_last),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(workers) > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )


def dataset_labels(dataset):
    imagefolder = getattr(dataset, "imagefolder", None)
    if imagefolder is not None and hasattr(imagefolder, "targets"):
        return list(imagefolder.targets)
    return [int(dataset[i][1]) for i in range(len(dataset))]


def calibration_split_indices(targets, frac: float, seed: int):
    """Class-stratified training-only split used for RMS calibration."""
    y = np.asarray(targets)
    rng = np.random.RandomState(int(seed))
    fit_idx, held_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c).copy()
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * float(frac))))
        held_idx += idx[:n].tolist()
        fit_idx += idx[n:].tolist()
    return np.asarray(sorted(fit_idx)), np.asarray(sorted(held_idx))


def append_csv(path: Path, row: Dict[str, Any]):
    first = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_score = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -float("inf"))
    best_other = masked.max(dim=1).values
    return true_score - best_other


def joint_losses(out, labels, confuser_map, args):
    source_logits = out["source_logits"].detach()
    final_logits = out["final_logits"]
    zero = final_logits.sum() * 0.0

    final_ce = F.cross_entropy(
        final_logits, labels, label_smoothing=args.label_smoothing
    )
    ento_ce = F.cross_entropy(
        out["ento_logits"], labels, label_smoothing=args.label_smoothing
    )

    # Training-only OOF hard-negative margin. Because the final model operates
    # over all classes, confuser classes are indexed directly in full logits.
    confuser_class, confuser_valid = choose_oof_confuser(labels, confuser_map)
    if confuser_valid.any():
        true_scores = final_logits.gather(1, labels[:, None]).squeeze(1)
        safe_confuser = confuser_class.clamp_min(0)
        confuser_scores = final_logits.gather(
            1, safe_confuser[:, None]
        ).squeeze(1)
        confuser_loss = F.relu(
            args.confuser_margin
            - true_scores[confuser_valid]
            + confuser_scores[confuser_valid]
        ).mean()
    else:
        confuser_loss = zero

    source_pred = source_logits.argmax(dim=1)
    source_correct = source_pred.eq(labels)
    source_margin = class_margin(source_logits, labels).detach()
    final_margin = class_margin(final_logits, labels)

    if source_correct.any():
        preserve_loss = F.relu(
            source_margin[source_correct] - final_margin[source_correct]
        ).mean()
    else:
        preserve_loss = zero

    source_error = ~source_correct
    if source_error.any():
        repair_loss = F.relu(
            args.repair_margin - final_margin[source_error]
        ).mean()
    else:
        repair_loss = zero

    overlap = region_overlap_loss(out["ento_heatmaps"])
    residual_l2 = out["residual_delta"].float().square().mean()

    total = (
        final_ce
        + args.ento_aux_weight * ento_ce
        + args.confuser_weight * confuser_loss
        + args.preserve_weight * preserve_loss
        + args.repair_weight * repair_loss
        + args.region_overlap_weight * overlap
        + args.residual_l2_weight * residual_l2
    )
    return {
        "total": total,
        "final_ce": final_ce,
        "ento_ce": ento_ce,
        "confuser": confuser_loss,
        "preserve": preserve_loss,
        "repair": repair_loss,
        "overlap": overlap,
        "residual_l2": residual_l2,
    }


def configure_trainable(model):
    for p in model.parameters():
        p.requires_grad_(False)

    names, params = [], []
    for name, p in model.named_parameters():
        if name.startswith("ento_") and name != "ento_scale_factor":
            p.requires_grad_(True)
            names.append(name)
            params.append(p)

    forbidden = [
        name for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("ento_")
    ]
    if forbidden:
        raise RuntimeError(f"Non-ENTO parameters remain trainable: {forbidden[:20]}")
    if not params:
        raise RuntimeError("No ENTO parameters were selected.")
    return names, params


def enforce_frozen_eval(model):
    model.core.eval()
    model.mvl_adapter.eval()


def train_epoch(model, loader, optimizer, scaler, device, confuser_map, args):
    model.train()
    enforce_frozen_eval(model)
    optimizer.zero_grad(set_to_none=True)

    keys = [
        "total", "final_ce", "ento_ce", "confuser",
        "preserve", "repair", "overlap", "residual_l2",
    ]
    sums = {k: 0.0 for k in keys}
    n_total = 0

    for step, batch in enumerate(loader, 1):
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast_context(args.amp and device.type == "cuda"):
            out = model(images)
            losses = joint_losses(out, labels, confuser_map, args)
            loss = losses["total"] / args.accum_steps

        scaler.scale(loss).backward()

        if step % args.accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        n = int(labels.numel())
        n_total += n
        for k in keys:
            sums[k] += float(losses[k].detach()) * n

    return {k: v / max(n_total, 1) for k, v in sums.items()}


@torch.no_grad()
def fit_scale_norm(model, loader, device, amp):
    """Fit the fixed ENTO residual RMS calibration factor using train images.

    E    = ENTO-like residual values across all classes.
    Mref = alpha_M * frozen MVL-branch logits, used only as a scale reference.

    gamma = RMS(E + Mref) / RMS(E/RMS(E) + Mref/RMS(Mref))
    s_E   = gamma / RMS(E)
    """
    model.eval()
    e_values, m_values = [], []

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        with autocast_context(amp and device.type == "cuda"):
            core = model.core(images, compute_ento=False)
            frozen = {
                "f2": core["f2"].detach(),
                "source_logits": core["logits"].detach(),
                "z_mvl_source": core["z_mvl"].detach(),
            }
            branch = model.branch_forward_from_features(frozen)
            e = branch["ento_delta"]
            alpha_m = model.core.anchor.alpha_mvl.detach()
            mref = alpha_m * core["logits_mvl"]

        e_values.append(e.float().cpu())
        m_values.append(mref.float().cpu())

    E = torch.cat(e_values, dim=0).double().reshape(-1)
    M = torch.cat(m_values, dim=0).double().reshape(-1)

    def rms(x):
        return float(torch.sqrt(torch.mean(x.square())).item())

    rms_e = rms(E)
    rms_m = rms(M)
    if rms_e <= 1e-12:
        raise RuntimeError("ENTO residual RMS is zero; calibration cannot be fitted.")
    if rms_m <= 1e-12:
        raise RuntimeError("MVL reference RMS is zero; calibration cannot be fitted.")

    raw_joint_rms = rms(E + M)
    unit_joint = E / rms_e + M / rms_m
    unit_joint_rms = rms(unit_joint)
    if unit_joint_rms <= 1e-12:
        raise RuntimeError("Unit-joint RMS is zero.")

    gamma = raw_joint_rms / unit_joint_rms
    s_e = gamma / rms_e

    return {
        "value_count": int(E.numel()),
        "rms_ento_raw": rms_e,
        "rms_source_mvl_reference_raw": rms_m,
        "raw_joint_rms": raw_joint_rms,
        "unit_joint_rms_before_gamma": unit_joint_rms,
        "target_branch_rms_gamma": gamma,
        "ento_scale_factor_sE": s_e,
        "rms_ento_after_scale": s_e * rms_e,
        "effective_alpha_times_ento_rms": float(
            model.residual_strength * s_e * rms_e
        ),
        "source_mvl_reference_definition":
            "alpha_M * frozen Compact-MVL logits_mvl across all classes",
        "formula":
            "gamma=RMS(E+Mref)/RMS(E/RMS(E)+Mref/RMS(Mref)); "
            "sE=gamma/RMS(E)",
        "labels_used_in_scale_values": False,
    }


def exact_mcnemar_p(corrected: int, degraded: int):
    n = int(corrected + degraded)
    if n == 0:
        return 1.0
    k = min(int(corrected), int(degraded))
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * p))


def paired_metrics(ref, pred, y):
    rc = ref.eq(y)
    pc = pred.eq(y)
    corrected = int(((~rc) & pc).sum())
    degraded = int((rc & (~pc)).sum())
    return {
        "top1": float(pc.float().mean() * 100.0),
        "correct_count": int(pc.sum()),
        "corrected": corrected,
        "degraded": degraded,
        "net": corrected - degraded,
        "changed": int(ref.ne(pred).sum()),
        "mcnemar_exact_p_vs_source": exact_mcnemar_p(corrected, degraded),
    }


@torch.no_grad()
def evaluate_once(model, loader, device, amp):
    """One official-test pass after the fixed epoch-12 training schedule."""
    model.eval()
    ys, source_preds, raw_preds, scaled_preds, aux_preds = [], [], [], [], []
    source_top5_hits, raw_top5_hits, scaled_top5_hits = [], [], []

    s_e = float(model.ento_scale_factor.detach().cpu())
    alpha = float(model.residual_strength)

    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast_context(amp and device.type == "cuda"):
            out = model(images)

        source_logits = out["source_logits"]
        ento_delta = out["ento_delta"]
        raw_logits = source_logits + alpha * ento_delta
        scaled_logits = out["final_logits"]

        src_pred = source_logits.argmax(dim=1)
        raw_pred = raw_logits.argmax(dim=1)
        scaled_pred = scaled_logits.argmax(dim=1)

        ys.append(labels.cpu())
        source_preds.append(src_pred.cpu())
        raw_preds.append(raw_pred.cpu())
        scaled_preds.append(scaled_pred.cpu())
        aux_preds.append(out["ento_logits"].argmax(1).cpu())

        # Top-5 here is an evaluation metric only, not an architectural mask.
        source_top5_hits.append(
            source_logits.topk(5, dim=1).indices.eq(labels[:, None]).any(1).cpu()
        )
        raw_top5_hits.append(
            raw_logits.topk(5, dim=1).indices.eq(labels[:, None]).any(1).cpu()
        )
        scaled_top5_hits.append(
            scaled_logits.topk(5, dim=1).indices.eq(labels[:, None]).any(1).cpu()
        )

    y = torch.cat(ys)
    src = torch.cat(source_preds)
    raw = torch.cat(raw_preds)
    scaled = torch.cat(scaled_preds)
    aux = torch.cat(aux_preds)
    src5 = torch.cat(source_top5_hits)
    raw5 = torch.cat(raw_top5_hits)
    scaled5 = torch.cat(scaled_top5_hits)

    source_top1 = float(src.eq(y).float().mean() * 100.0)
    raw_m = paired_metrics(src, raw, y)
    scaled_m = paired_metrics(src, scaled, y)

    return {
        "count": int(y.numel()),
        "source": {
            "top1": source_top1,
            "correct_count": int(src.eq(y).sum()),
            "top5": float(src5.float().mean() * 100.0),
        },
        "joint_without_scale_calibration": {
            **raw_m,
            "top5": float(raw5.float().mean() * 100.0),
            "gain_vs_source_pp": raw_m["top1"] - source_top1,
            "formula": "M + alpha * DeltaE across all classes",
            "sE": 1.0,
        },
        "ento_mvl_final": {
            **scaled_m,
            "top5": float(scaled5.float().mean() * 100.0),
            "gain_vs_source_pp": scaled_m["top1"] - source_top1,
            "formula": "M + alpha * sE * DeltaE across all classes",
            "sE": s_e,
        },
        "ento_aux_top1": float(aux.eq(y).float().mean() * 100.0),
        "official_variant_selection": False,
    }


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def audit_source_checkpoint(path: Path, requested_seed: int):
    ck = safe_load(path)
    source_seed = ck.get("seed", ck.get("args", {}).get("seed", None))
    if source_seed is not None and int(source_seed) != int(requested_seed):
        raise RuntimeError(
            f"Matched-seed violation: requested={requested_seed}, "
            f"source checkpoint seed={source_seed}"
        )
    return {
        "source_seed": None if source_seed is None else int(source_seed),
        "weights_kind": ck.get("weights_kind", None),
        "ema_smoothing_used": ck.get("ema_smoothing_used", None),
        "ema_decay_requested": ck.get(
            "ema_decay_requested", ck.get("args", {}).get("ema_decay", None)
        ),
    }


def infer_or_resolve_joint_seed(seed, source_checkpoint: Path) -> int:
    if seed is not None:
        return resolve_seed(seed, "ENTO+MVL")
    ck = safe_load(source_checkpoint)
    source_seed = ck.get("seed", ck.get("args", {}).get("seed", None))
    if source_seed is not None:
        print(f"[SEED] ENTO+MVL: inferred matched source seed={int(source_seed)}")
        return int(source_seed)
    return resolve_seed(None, "ENTO+MVL")


def main():
    args = parse_args()
    source_path = Path(args.source_checkpoint).resolve()
    args.seed = infer_or_resolve_joint_seed(args.seed, source_path)
    out = Path(args.output_dir).resolve()

    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)

    confuser_path = Path(args.confuser_map).resolve()
    source_audit = audit_source_checkpoint(source_path, args.seed)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    # Final ENTO adaptation disables random erasing, matching the manuscript.
    train_set, test_set, attribute_names = build_attribute_datasets(
        data_root=args.data_root,
        cub_root=args.cub_root,
        img_size=args.img_size,
        random_erasing=False,
        attribute_mode="basic",
    )
    train_eval_set, _, _ = build_attribute_datasets(
        data_root=args.data_root,
        cub_root=args.cub_root,
        img_size=args.img_size,
        random_erasing=False,
        attribute_mode="basic",
    )

    _, eval_tf = build_transforms(img_size=args.img_size, random_erasing=False)
    train_eval_set.imagefolder.transform = eval_tf
    test_set.imagefolder.transform = eval_tf

    if len(train_set) != 5994:
        raise RuntimeError(f"Expected train=5994, got {len(train_set)}")
    if len(test_set) != 5794:
        raise RuntimeError(f"Expected official eval=5794, got {len(test_set)}")

    labels = dataset_labels(train_set)
    scale_idx, excluded_idx = calibration_split_indices(
        labels, args.scale_calibration_fraction, args.scale_split_seed
    )
    if len(scale_idx) != 5394 or len(excluded_idx) != 600:
        raise RuntimeError(
            f"Expected RMS-calibration split 5394/600, got "
            f"{len(scale_idx)}/{len(excluded_idx)}"
        )

    train_loader = make_loader(
        train_set, args.batch_size, True, True, args.num_workers, args.seed
    )
    scale_loader = make_loader(
        Subset(train_eval_set, scale_idx.tolist()),
        max(args.batch_size, 8), False, False, args.num_workers, args.seed + 30000,
    )
    official_loader = make_loader(
        test_set,
        max(args.batch_size, 4), False, False, args.num_workers, args.seed + 50000,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confuser_map = load_confuser_map(str(confuser_path), len(train_set.classes))
    if confuser_map is None:
        raise RuntimeError("OOF confuser map could not be loaded.")

    num_classes = len(train_set.classes)
    model = ENTOMVL(
        num_classes=num_classes,
        num_attributes=len(attribute_names),
        neural_mode="no_neural",
        embed_dim=512,
        num_regions=args.num_regions,
        region_dim=args.region_dim,
        branch_dim=args.branch_dim,
        mvl_adapter_bottleneck=256,
        residual_hidden_dim=args.residual_hidden_dim,
        residual_dropout=args.residual_dropout,
        residual_logit_clip=args.residual_logit_clip,
        residual_strength=args.alpha_e,
        ento_scale_factor=1.0,
    )

    load = model.load_compact_mvl(str(source_path), prefer_ema=False)
    model.freeze_source()
    trainable_names, trainable_params = configure_trainable(model)
    model.to(device)
    enforce_frozen_eval(model)

    if model.enable_mvl_adapter:
        raise RuntimeError("The final joint model must not add an extra MVL residual.")
    if hasattr(model, "interaction"):
        raise RuntimeError("Unexpected interaction module in final architecture.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)

    print("=" * 132)
    print("FINAL ENTO-MVL | ALL-CLASS JOINT INTEGRATION")
    print(f"seed={args.seed} | source_seed={source_audit['source_seed']}")
    print(
        f"{args.img_size}x{args.img_size} | epochs={args.epochs} | "
        f"batch={args.batch_size} | accum={args.accum_steps}"
    )
    print("M = z_base + alpha_M*z_M (frozen trained MVL-like source)")
    print("z_A = M + 0.20*s_E*Delta_E for every class")
    print("MVL-like source frozen | ENTO-like residual pathway trainable")
    print(
        "Auxiliary training terms: ENTO classification + OOF confuser margin + "
        "source-margin preservation + error repair + region overlap + residual L2"
    )
    print(
        "RMS calibration: 5394 training images; official 5794 test images are "
        "not used for calibration or checkpoint selection"
    )
    print(
        f"OOF confuser={confuser_path} | load missing={len(load.missing_keys)} "
        f"unexpected={len(load.unexpected_keys)}"
    )
    print(f"params total={total_params:,} | trainable ENTO={trainable_count:,}")
    print("=" * 132)

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = make_scaler(args.amp and device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(
            model, train_loader, optimizer, scaler, device, confuser_map, args
        )
        row = {"epoch": epoch, **tr}
        append_csv(out / "train_log.csv", row)
        print(
            f"ENTO-MVL | {epoch:03d}/{args.epochs} | "
            f"loss={tr['total']:.5f} | CE={tr['final_ce']:.5f} | "
            f"Eaux={tr['ento_ce']:.5f} | Conf={tr['confuser']:.5f} | "
            f"Preserve={tr['preserve']:.5f} | Repair={tr['repair']:.5f} | "
            f"Overlap={tr['overlap']:.5f} | R2={tr['residual_l2']:.6f}"
        )

    # Prespecified final epoch; no validation/test checkpoint selection.
    model.eval()
    scale = fit_scale_norm(model, scale_loader, device, args.amp)
    model.set_ento_scale_factor(scale["ento_scale_factor_sE"])
    (out / "scale_norm.json").write_text(
        json.dumps(scale, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    checkpoint = {
        "epoch": args.epochs,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classes": list(train_set.classes),
        "attribute_names": list(attribute_names),
        "seed": args.seed,
        "source_seed": source_audit["source_seed"],
        "source_checkpoint": str(source_path),
        "confuser_map": str(confuser_path),
        "architecture": "ENTO-MVL all-class residual integration",
        "formula": "z_A = M + alpha*s_E*Delta_E for all classes",
        "interaction": False,
        "extra_deltaM": False,
        "alpha_E": args.alpha_e,
        "all_class_residual": True,
        "num_classes": num_classes,
        "s_E": scale["ento_scale_factor_sE"],
        "scale_norm": scale,
        "scale_fit_count": len(scale_idx),
        "scale_excluded_count": len(excluded_idx),
        "scale_split_seed": args.scale_split_seed,
        "official_test_used_for_training": False,
        "official_test_used_for_checkpoint_selection": False,
        "official_test_used_for_scale_fit": False,
        "checkpoint_selection":
            "fixed final epoch 12; no official/validation selection",
        "args": vars(args),
    }
    torch.save(checkpoint, out / "final_ento_mvl.pth")

    official = evaluate_once(model, official_loader, device, args.amp)
    summary = {
        "experiment": "ento_mvl_all_class_v1",
        "seed": args.seed,
        "source_seed": source_audit["source_seed"],
        "formula": checkpoint["formula"],
        "interaction": False,
        "extra_deltaM": False,
        "alpha_E": args.alpha_e,
        "all_class_residual": True,
        "num_classes": num_classes,
        "s_E": scale["ento_scale_factor_sE"],
        "scale_norm": scale,
        "source_checkpoint": str(source_path),
        "source_audit": source_audit,
        "oof_confuser_map": str(confuser_path),
        "fixed_epoch": args.epochs,
        "official_test_used_for_training_or_selection": False,
        "official_test_used_for_scale_fit": False,
        "official": official,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count": total_params,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 132)
    print("RMS CALIBRATION")
    print(json.dumps(scale, indent=2, ensure_ascii=False))
    print("\nOFFICIAL 5794 — ONE PASS, NO VARIANT SELECTION")
    print(json.dumps(official, indent=2, ensure_ascii=False))
    print("=" * 132)
    print(f"Saved: {out / 'final_ento_mvl.pth'}")
    print(f"Saved: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
