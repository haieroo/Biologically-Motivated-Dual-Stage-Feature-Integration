# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.resolve().parents[1]
CORE = REPO_ROOT / "core"
sys.path.insert(0, str(CORE))

from data_pipeline import build_datasets
from models import CleanResNet50, load_checkpoint_state
from models_compact_mvl import CompactMVLResNet50
from models_single_region_branch import IndependentAutoRegionBranchResNet50
from models_ento_mvl import ENTOMVL


def parse_args():
    p = argparse.ArgumentParser(
        "Unified 3-seed four-model ablation evaluation."
    )
    p.add_argument("--project-root", required=True)
    p.add_argument("--clean-checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--seeds", default="", help="Optional. If omitted, auto-discover results/seed* runs.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_historical_amp():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def amp_ctx(enabled):
    if torch.cuda.is_available():
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=enabled)
    from contextlib import nullcontext
    return nullcontext()


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_clean_model(path: Path, n_classes: int, device):
    model = CleanResNet50(num_classes=n_classes, pretrained=False)
    _, state = load_checkpoint_state(str(path), prefer_ema=False)
    current = model.state_dict()
    transfer = {
        k: v for k, v in state.items()
        if k in current and current[k].shape == v.shape
    }
    missing = [k for k in current if k not in transfer]
    if missing:
        raise RuntimeError(
            f"Clean checkpoint missing model keys: {missing[:20]}"
        )
    result = model.load_state_dict(transfer, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_ento_model(path: Path, device):
    ck = safe_load(path)
    args = dict(ck.get("args", {}))
    classes = ck.get("classes") or []
    model = IndependentAutoRegionBranchResNet50(
        num_classes=len(classes) or 200,
        region="ento",
        num_regions=int(args.get("num_regions", 4)),
        region_dim=int(args.get("region_dim", 256)),
        branch_dim=int(args.get("branch_dim", 512)),
        dropout=float(args.get("dropout", 0.10)),
    )
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def load_mvl_model(path: Path, device):
    ck = safe_load(path)
    classes = ck.get("classes") or []
    args = dict(ck.get("args", {}))
    mode = str(ck.get("ablation_mode", "compact_mvl"))
    model = CompactMVLResNet50(
        num_classes=len(classes) or 200,
        mode=mode,
        pretrained=False,
        embed_dim=int(args.get("embed_dim", 512)),
        dropout=float(args.get("dropout", 0.20)),
    )
    state = ck.get("model")
    if state is None:
        state = ck.get("raw_model")
    if state is None:
        raise RuntimeError(f"No model/raw_model in {path}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def load_joint_model(path: Path, device):
    ck = safe_load(path)
    args = dict(ck.get("args", {}))
    classes = ck.get("classes") or []
    attrs = ck.get("attribute_names") or []
    model = ENTOMVL(
        num_classes=len(classes) or 200,
        num_attributes=len(attrs),
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


def exact_mcnemar(corrected: int, degraded: int):
    n = corrected + degraded
    if n == 0:
        return 1.0
    k = min(corrected, degraded)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * p))


def metrics(pred, top5, labels):
    correct = pred.eq(labels)
    return {
        "top1": 100.0 * float(correct.float().mean()),
        "top5": 100.0 * float(
            top5.eq(labels[:, None]).any(1).float().mean()
        ),
        "correct_count": int(correct.sum()),
        "count": int(labels.numel()),
    }


def paired(ref_pred, pred, labels):
    rc = ref_pred.eq(labels)
    pc = pred.eq(labels)
    corrected = int(((~rc) & pc).sum())
    degraded = int((rc & (~pc)).sum())
    return {
        "changed": int(ref_pred.ne(pred).sum()),
        "corrected": corrected,
        "degraded": degraded,
        "net": corrected - degraded,
        "mcnemar_exact_p": exact_mcnemar(corrected, degraded),
    }


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@torch.inference_mode()
def evaluate_all(clean, ento, mvl, joint, loader, device):
    ys = []
    pred = {k: [] for k in ["ResNet50", "ENTO-like", "MVL-like", "ENTO+MVL"]}
    top5 = {k: [] for k in pred}
    ento_base_pred = []
    ento_base_logit_maxdiff = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_ctx(device.type == "cuda"):
            c = clean(images)
            e = ento(images)
            m = mvl(images)
            j = joint(images)

        c_logits = c["logits"] if isinstance(c, dict) else c
        e_logits = e["logits"]
        m_logits = m["logits"]
        j_logits = j["final_logits"]

        ento_base_logit_maxdiff = max(
            ento_base_logit_maxdiff,
            float(
                (c_logits.float() - e["logits_base"].float())
                .abs().max().item()
            ),
        )

        logits_map = {
            "ResNet50": c_logits,
            "ENTO-like": e_logits,
            "MVL-like": m_logits,
            "ENTO+MVL": j_logits,
        }
        for name, logits in logits_map.items():
            if name == "ENTO+MVL":
                p = j["full_prediction"]
            else:
                p = logits.argmax(1)
            pred[name].append(p.cpu())
            top5[name].append(
                logits.topk(min(5, logits.size(1)), 1).indices.cpu()
            )

        ento_base_pred.append(e["logits_base"].argmax(1).cpu())
        ys.append(labels.cpu())

    y = torch.cat(ys)
    pred = {k: torch.cat(v) for k, v in pred.items()}
    top5 = {k: torch.cat(v) for k, v in top5.items()}
    ento_base_pred = torch.cat(ento_base_pred)

    out = {
        name: metrics(pred[name], top5[name], y)
        for name in pred
    }
    out["audit"] = {
        "clean_vs_ento_embedded_base_prediction_mismatch":
            int(pred["ResNet50"].ne(ento_base_pred).sum()),
        "clean_vs_ento_embedded_base_logits_max_abs":
            ento_base_logit_maxdiff,
    }
    out["pairs"] = {
        "ENTO-like_vs_ResNet50":
            paired(pred["ResNet50"], pred["ENTO-like"], y),
        "MVL-like_vs_ResNet50":
            paired(pred["ResNet50"], pred["MVL-like"], y),
        "ENTO+MVL_vs_MVL-like":
            paired(pred["MVL-like"], pred["ENTO+MVL"], y),
        "ENTO+MVL_vs_ResNet50":
            paired(pred["ResNet50"], pred["ENTO+MVL"], y),
    }
    return out, y, pred



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
    data_root = Path(a.data_root).resolve()
    seeds = ([int(x.strip()) for x in a.seeds.split(",") if x.strip()] if str(a.seeds).strip() else discover_seeds(project))
    outdir = (
        Path(a.output_dir).resolve()
        if a.output_dir
        else project / "results" / "formal_four_model_ablation_3seeds"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    clean_path = Path(a.clean_checkpoint).resolve()
    if not clean_path.exists():
        raise FileNotFoundError(clean_path)

    _, eval_set = build_datasets(
        str(data_root),
        img_size=a.img_size,
        random_erasing=False,
    )
    if len(eval_set) != 5794:
        raise RuntimeError(
            f"Expected 5794 official images, got {len(eval_set)}"
        )

    configure_historical_amp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean = load_clean_model(clean_path, len(eval_set.classes), device)

    rows = []
    full = []

    print("=" * 146)
    print("FORMAL 3-SEED FOUR-MODEL ABLATION | SAME 5794 EVALUATOR")
    print("Canonical inference: historical_amp")
    print("ResNet50 = clean logits")
    print("ENTO-like = z_base + alpha_E*z_E")
    print("MVL-like = z_base + alpha_M*z_M")
    print("ENTO+MVL = frozen MVL-like output + calibrated ENTO-like residual across all classes")
    print("=" * 146)

    for seed in seeds:
        set_seed(seed)
        ento_path = (
            project / "results" / f"seed{seed}"
            / "ento_like_formal_fixed12" / "final_epoch12.pth"
        )
        mvl_path = (
            project / "results" / f"seed{seed}"
            / "compact_mvl_ema0" / "best_raw_only.pth"
        )
        joint_path = (
            project / "results" / f"seed{seed}"
            / "ento_mvl" / "final_ento_mvl.pth"
        )
        for p in [ento_path, mvl_path, joint_path]:
            if not p.exists():
                raise FileNotFoundError(p)

        ento, ento_ck = load_ento_model(ento_path, device)
        mvl, mvl_ck = load_mvl_model(mvl_path, device)
        joint, joint_ck = load_joint_model(joint_path, device)

        # Check seed provenance.
        if int(ento_ck.get("seed", -1)) != seed:
            raise RuntimeError(f"ENTO seed mismatch for {seed}")
        mvl_seed = mvl_ck.get("seed", mvl_ck.get("args", {}).get("seed", None))
        if mvl_seed is not None and int(mvl_seed) != seed:
            raise RuntimeError(f"MVL seed mismatch for {seed}: {mvl_seed}")
        if int(joint_ck.get("seed", -1)) != seed:
            raise RuntimeError(f"Joint seed mismatch for {seed}")

        g = torch.Generator().manual_seed(seed + 99000)
        loader = DataLoader(
            eval_set,
            batch_size=a.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=a.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=a.num_workers > 0,
            generator=g,
        )

        result, y, preds = evaluate_all(
            clean, ento, mvl, joint, loader, device
        )

        base = result["ResNet50"]["top1"]
        mvl_top1 = result["MVL-like"]["top1"]

        for method in ["ResNet50", "ENTO-like", "MVL-like", "ENTO+MVL"]:
            m = result[method]
            total_params = {
                "ResNet50": sum(p.numel() for p in clean.parameters()),
                "ENTO-like": sum(p.numel() for p in ento.parameters()),
                "MVL-like": sum(p.numel() for p in mvl.parameters()),
                "ENTO+MVL": sum(p.numel() for p in joint.parameters()),
            }[method]
            row = {
                "seed": seed,
                "method": method,
                "top1": m["top1"],
                "top5": m["top5"],
                "correct_count": m["correct_count"],
                "n": m["count"],
                "gain_vs_ResNet50_pp": m["top1"] - base,
                "gain_vs_MVL_pp": (
                    m["top1"] - mvl_top1
                    if method == "ENTO+MVL"
                    else ""
                ),
                "params": int(total_params),
                "alpha_E": (
                    float(ento.alpha.detach().cpu())
                    if method == "ENTO-like"
                    else ""
                ),
                "alpha_M": (
                    float(mvl.alpha_mvl.detach().cpu())
                    if method == "MVL-like"
                    else ""
                ),
                "sE": (
                    float(joint.ento_scale_factor.detach().cpu())
                    if method == "ENTO+MVL"
                    else ""
                ),
            }
            rows.append(row)

        seed_result = {
            "seed": seed,
            "checkpoints": {
                "clean": str(clean_path),
                "ento": str(ento_path),
                "mvl": str(mvl_path),
                "joint": str(joint_path),
            },
            "metrics": result,
            "formulae": {
                "ResNet50": "z_base",
                "ENTO-like": "z_base + alpha_E * z_E(logits)",
                "MVL-like": "z_base + alpha_M * z_M(logits)",
                "ENTO+MVL": (
                    "M=z_base+alpha_M*z_M; "
                    "z_A=M+0.20*sE*DeltaE for all classes"
                ),
            },
            "ento_alpha_E": float(ento.alpha.detach().cpu()),
            "mvl_alpha_M": float(mvl.alpha_mvl.detach().cpu()),
            "joint_sE": float(joint.ento_scale_factor.detach().cpu()),
        }
        full.append(seed_result)
        (outdir / f"seed{seed}_ablation.json").write_text(
            json.dumps(seed_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Per-image audit.
        image_rows = []
        samples = eval_set.samples
        for i in range(len(y)):
            image_rows.append({
                "index": i,
                "path": samples[i][0],
                "true_label": int(y[i]),
                "ResNet50_pred": int(preds["ResNet50"][i]),
                "ENTO_like_pred": int(preds["ENTO-like"][i]),
                "MVL_like_pred": int(preds["MVL-like"][i]),
                "ENTO_MVL_pred": int(preds["ENTO+MVL"][i]),
            })
        write_csv(outdir / f"seed{seed}_predictions.csv", image_rows)

        print(
            f"seed={seed} | "
            f"Base={result['ResNet50']['top1']:.6f} | "
            f"ENTO={result['ENTO-like']['top1']:.6f} "
            f"({result['ENTO-like']['top1']-base:+.6f}) | "
            f"MVL={result['MVL-like']['top1']:.6f} "
            f"({result['MVL-like']['top1']-base:+.6f}) | "
            f"Joint={result['ENTO+MVL']['top1']:.6f} "
            f"(vs MVL {result['ENTO+MVL']['top1']-mvl_top1:+.6f}) | "
            f"Clean↔ENTO-base mismatch="
            f"{result['audit']['clean_vs_ento_embedded_base_prediction_mismatch']}"
        )

        del ento, mvl, joint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(outdir / "01_per_seed_four_model_ablation.csv", rows)

    methods = ["ResNet50", "ENTO-like", "MVL-like", "ENTO+MVL"]
    agg_rows = []
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        top1 = np.asarray([float(r["top1"]) for r in rr])
        top5 = np.asarray([float(r["top5"]) for r in rr])
        gain = np.asarray([float(r["gain_vs_ResNet50_pp"]) for r in rr])
        agg_rows.append({
            "method": method,
            "n_seeds": len(rr),
            "top1_mean": float(top1.mean()),
            "top1_sd": float(top1.std(ddof=1)) if len(rr) > 1 else 0.0,
            "top5_mean": float(top5.mean()),
            "top5_sd": float(top5.std(ddof=1)) if len(rr) > 1 else 0.0,
            "gain_vs_ResNet50_mean_pp": float(gain.mean()),
            "gain_vs_ResNet50_sd_pp": (
                float(gain.std(ddof=1)) if len(rr) > 1 else 0.0
            ),
        })
    write_csv(outdir / "02_four_model_mean_sd.csv", agg_rows)

    paired_ento = np.asarray([
        next(r["top1"] for r in rows if r["seed"] == s and r["method"] == "ENTO-like")
        - next(r["top1"] for r in rows if r["seed"] == s and r["method"] == "ResNet50")
        for s in seeds
    ], dtype=float)
    paired_mvl = np.asarray([
        next(r["top1"] for r in rows if r["seed"] == s and r["method"] == "MVL-like")
        - next(r["top1"] for r in rows if r["seed"] == s and r["method"] == "ResNet50")
        for s in seeds
    ], dtype=float)
    paired_joint_vs_mvl = np.asarray([
        next(r["top1"] for r in rows if r["seed"] == s and r["method"] == "ENTO+MVL")
        - next(r["top1"] for r in rows if r["seed"] == s and r["method"] == "MVL-like")
        for s in seeds
    ], dtype=float)

    summary = {
        "protocol": "formal_current_four_model_ablation_v1",
        "canonical_evaluator": "historical_amp",
        "seeds": seeds,
        "official_eval_count": len(eval_set),
        "formal_ento_training": {
            "source": "same Clean ResNet50 checkpoint",
            "training_images": 5994,
            "epochs": 12,
            "checkpoint_selection": "fixed epoch 12",
            "neural_rdm_loss": 0.0,
            "attribute_part_bbox_supervision": False,
            "formula": "z_base + alpha_E * z_E(logits)",
        },
        "mean_sd": agg_rows,
        "paired_gains": {
            "ENTO_like_minus_ResNet50_pp": paired_ento.tolist(),
            "ENTO_like_minus_ResNet50_mean_pp": float(paired_ento.mean()),
            "ENTO_like_minus_ResNet50_sd_pp": float(paired_ento.std(ddof=1)),
            "MVL_like_minus_ResNet50_pp": paired_mvl.tolist(),
            "MVL_like_minus_ResNet50_mean_pp": float(paired_mvl.mean()),
            "MVL_like_minus_ResNet50_sd_pp": float(paired_mvl.std(ddof=1)),
            "ENTO_MVL_minus_MVL_like_pp": paired_joint_vs_mvl.tolist(),
            "ENTO_MVL_minus_MVL_like_mean_pp": float(paired_joint_vs_mvl.mean()),
            "ENTO_MVL_minus_MVL_like_sd_pp": float(paired_joint_vs_mvl.std(ddof=1)),
        },
        "per_seed": full,
        "important_note": (
            "The formal ENTO-like row is newly retrained on the full 5994 training set "
            "with fixed epoch 12 and official 5794 evaluated only here. It should not be "
            "mixed with the older 50/50 calibration-selected ENTO-like reproduction."
        ),
    }
    (outdir / "03_four_model_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 146)
    print("THREE-SEED MEAN ± SD")
    for r in agg_rows:
        print(
            f"{r['method']:12s} | "
            f"Top1={r['top1_mean']:.6f} ± {r['top1_sd']:.6f} | "
            f"Top5={r['top5_mean']:.6f} ± {r['top5_sd']:.6f} | "
            f"ΔBase={r['gain_vs_ResNet50_mean_pp']:+.6f} ± "
            f"{r['gain_vs_ResNet50_sd_pp']:.6f}"
        )
    print("=" * 146)
    print("[ALL DONE]")
    print(outdir / "01_per_seed_four_model_ablation.csv")
    print(outdir / "02_four_model_mean_sd.csv")
    print(outdir / "03_four_model_ablation_summary.json")


if __name__ == "__main__":
    main()
