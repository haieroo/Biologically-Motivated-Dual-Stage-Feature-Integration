from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE = REPO_ROOT / "core"
sys.path.insert(0, str(CORE))

from seed_utils import parse_seed_list, save_manifest

PAPER_SEEDS = [666, 2026, 7777]


def run(cmd):
    cmd = [str(x) for x in cmd]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(
        "Reproduce the three-seed ENTO-MVL CUB-200-2011 experiment chain."
    )
    p.add_argument("--data-root", required=True)
    p.add_argument("--cub-root", required=True)
    p.add_argument("--clean-checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument(
        "--seeds",
        default="666,2026,7777",
        help="Comma-separated seeds. Defaults to the three manuscript seeds.",
    )
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def main():
    a = parse_args()
    outroot = Path(a.output_root).resolve()
    results = outroot / "results"
    results.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_list(a.seeds)
    if not seeds:
        seeds = PAPER_SEEDS.copy()

    save_manifest(
        outroot / "experiment_manifest.json",
        seeds,
        seed_source="manuscript_default" if seeds == PAPER_SEEDS else "user_supplied",
        num_runs=len(seeds),
        note=(
            "Each run uses one matched seed across MVL-like training, OOF hard-negative "
            "construction, standalone ENTO-like training, and the final all-class ENTO-MVL stage."
        ),
    )
    print(f"[RUNS] seeds={seeds}")

    R = REPO_ROOT
    for index, seed in enumerate(seeds, 1):
        print("\n" + "=" * 120)
        print(f"RUN {index}/{len(seeds)} | seed={seed}")
        print("=" * 120)

        sr = results / f"seed{seed}"

        # 1) MVL-like source model.
        mvl = sr / "mvl_like"
        run([
            a.python, R / "training" / "train_mvl_like.py",
            "--data_root", a.data_root,
            "--output_dir", mvl,
            "--clean_checkpoint", a.clean_checkpoint,
            "--ablation_mode", "compact_mvl",
            "--img_size", "448",
            "--batch_size", "2",
            "--accum_steps", "4",
            "--epochs", "50",
            "--plugin_warmup_epochs", "5",
            "--warmup_epochs", "2",
            "--plugin_lr", "2e-4",
            "--head_lr", "2e-5",
            "--layer4_lr", "8e-6",
            "--layer3_lr", "4e-6",
            "--weight_decay", "1e-4",
            "--aux_weight", ".30",
            "--label_smoothing", ".05",
            "--mix_prob", ".25",
            "--mixup_alpha", ".2",
            "--cutmix_alpha", ".5",
            "--grad_clip", ".5",
            "--ema_decay", "0.0",
            "--num_workers", "2",
            "--seed", seed,
            "--amp",
            "--overwrite",
        ])

        run([
            a.python, R / "tools" / "export_raw_mvl_checkpoint.py",
            "--checkpoint", mvl / "best.pth",
            "--output", mvl / "best_raw_only.pth",
            "--seed", seed,
        ])

        # 2) Training-only out-of-fold confuser map for the auxiliary margin loss.
        oof = sr / "oof"
        oof_csv = oof / "compact_mvl_oof_predictions.csv"
        oof_map = oof / "oof_confuser_map.json"
        run([
            a.python, R / "tools" / "generate_oof_predictions.py",
            "--data_root", a.data_root,
            "--cub_root", a.cub_root,
            "--output_dir", oof / "fold_training",
            "--output_csv", oof_csv,
            "--attribute_mode", "basic",
            "--img_size", "448",
            "--folds", "5",
            "--epochs", "12",
            "--batch_size", "2",
            "--accum_steps", "4",
            "--num_workers", "2",
            "--backbone_lr", "1e-5",
            "--head_lr", "1e-4",
            "--weight_decay", "1e-4",
            "--grad_clip", "1.0",
            "--label_smoothing", ".05",
            "--split_seed", "1731",
            "--seed", seed,
            "--amp",
            "--overwrite",
        ])
        run([
            a.python, R / "tools" / "build_oof_confuser_map.py",
            "--oof_csv", oof_csv,
            "--output_json", oof_map,
            "--num_classes", "200",
            "--neighbors", "5",
        ])

        # 3) Standalone ENTO-like ablation.
        ento = sr / "ento_like"
        run([
            a.python, R / "training" / "train_ento_like.py",
            "--data-root", a.data_root,
            "--cub-root", a.cub_root,
            "--clean-checkpoint", a.clean_checkpoint,
            "--output-dir", ento,
            "--seed", seed,
            "--img-size", "448",
            "--epochs", "12",
            "--batch-size", "2",
            "--accum-steps", "4",
            "--num-workers", "2",
            "--num-regions", "4",
            "--region-dim", "256",
            "--branch-dim", "512",
            "--dropout", ".10",
            "--lr", "1e-4",
            "--weight-decay", "1e-4",
            "--aux-weight", ".20",
            "--label-smoothing", ".05",
            "--grad-clip", ".5",
            "--amp",
            "--overwrite",
        ])

        # 4) Final ENTO-MVL: MVL-like source frozen, ENTO residual trained for 12 epochs.
        joint = sr / "ento_mvl"
        run([
            a.python, R / "training" / "train_ento_mvl.py",
            "--data-root", a.data_root,
            "--cub-root", a.cub_root,
            "--source-checkpoint", mvl / "best_raw_only.pth",
            "--confuser-map", oof_map,
            "--output-dir", joint,
            "--seed", seed,
            "--img-size", "448",
            "--epochs", "12",
            "--batch-size", "2",
            "--accum-steps", "4",
            "--num-workers", "2",
            "--num-regions", "4",
            "--region-dim", "256",
            "--branch-dim", "512",
            "--residual-hidden-dim", "256",
            "--residual-dropout", ".10",
            "--residual-logit-clip", "1.0",
            "--alpha-e", ".20",
            "--lr", "1e-4",
            "--weight-decay", "1e-4",
            "--grad-clip", ".5",
            "--label-smoothing", ".05",
            "--ento-aux-weight", ".20",
            "--confuser-weight", ".10",
            "--confuser-margin", ".20",
            "--preserve-weight", ".30",
            "--repair-weight", ".10",
            "--repair-margin", ".10",
            "--region-overlap-weight", ".01",
            "--residual-l2-weight", "1e-3",
            "--scale-calibration-fraction", ".10",
            "--scale-split-seed", "5204096",
            "--amp",
            "--overwrite",
        ])

    print("\n[ALL RUNS COMPLETE]")
    print(outroot / "experiment_manifest.json")


if __name__ == "__main__":
    main()
