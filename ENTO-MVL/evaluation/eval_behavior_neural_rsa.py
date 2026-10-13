# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from posthoc_common import (
    OrderedStimulusDataset,
    autocast_context,
    bh_fdr,
    build_joint_model,
    configure_historical_amp,
    load_neural_manifest,
    load_square_matrix,
    pearson_distance_rdm,
    set_seed,
    spearman_rdm,
    stimulus_label_delta_test,
    stimulus_label_permutation_test,
    write_csv,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(
        "Final 3-seed behavior/neural RSA for ENTO-MVL."
    )
    p.add_argument("--project-root", required=True)
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--seeds", default="", help="Optional. If omitted, auto-discover results/seed* runs.")
    p.add_argument("--manifest", default="")
    p.add_argument("--behavior-rdm", default="")
    p.add_argument("--ento-neural-rdm", default="")
    p.add_argument("--mvl-neural-rdm", default="")
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


@torch.no_grad()
def extract(model, loader, device):
    pieces = {
        "z_ento": [],
        "z_mvl": [],
        "z_joint_mean": [],
        "source_logits": [],
        "final_logits": [],
    }
    order = []
    model.eval()
    for images, idx, stimulus_id, _path in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(device.type == "cuda"):
            out = model(images)
        z_e = out["z_ento"].float()
        z_m = out["z_mvl"].float()
        pieces["z_ento"].append(z_e.cpu())
        pieces["z_mvl"].append(z_m.cpu())
        # Diagnostic representational readout only; NOT a classifier head.
        pieces["z_joint_mean"].append((0.5 * (z_e + z_m)).cpu())
        pieces["source_logits"].append(out["source_logits"].float().cpu())
        pieces["final_logits"].append(out["final_logits"].float().cpu())
        order.extend([int(x) for x in idx])
    if order != list(range(len(order))):
        raise RuntimeError("36-stimulus DataLoader order changed.")
    return {k: torch.cat(v, 0).numpy() for k, v in pieces.items()}


def mean_sd(values):
    values = [float(x) for x in values]
    if len(values) == 1:
        return values[0], 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))



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
    input_dir = legacy / "neural_rdm_inputs"
    manifest_path = Path(a.manifest) if a.manifest else input_dir / "36_neural_manifest.csv"
    behavior_path = Path(a.behavior_rdm) if a.behavior_rdm else input_dir / "Behavior_RDM.csv"
    ento_path = Path(a.ento_neural_rdm) if a.ento_neural_rdm else input_dir / "ENTO_crossnobis_RDM.csv"
    mvl_path = Path(a.mvl_neural_rdm) if a.mvl_neural_rdm else input_dir / "MVL_crossnobis_RDM.csv"
    outdir = (
        Path(a.output_dir).resolve()
        if a.output_dir
        else project / "results" / "final_behavior_rsa_3seeds"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    for p in [manifest_path, behavior_path, ento_path, mvl_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    rows_manifest = load_neural_manifest(str(manifest_path))
    behavior = load_square_matrix(str(behavior_path))
    ento_neural = load_square_matrix(str(ento_path))
    mvl_neural = load_square_matrix(str(mvl_path))

    n = len(rows_manifest)
    for name, rdm in [
        ("Behavior", behavior), ("ENTO neural", ento_neural), ("MVL neural", mvl_neural)
    ]:
        if rdm.shape != (n, n):
            raise RuntimeError(f"{name} RDM {rdm.shape} != manifest {n}x{n}")
    if n != 36:
        print(f"[WARNING] Expected 36 stimuli, found {n}. Analysis still runs.")

    seeds = ([int(x.strip()) for x in a.seeds.split(",") if x.strip()] if str(a.seeds).strip() else discover_seeds(project))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_historical_amp()

    dataset = OrderedStimulusDataset(rows_manifest, img_size=a.img_size)
    loader = DataLoader(
        dataset,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=a.num_workers > 0,
    )

    per_seed_rows: List[Dict] = []
    seed_rdms: Dict[int, Dict[str, np.ndarray]] = {}

    print("=" * 136)
    print("FINAL 3-SEED BEHAVIOR / NEURAL RSA")
    print(f"Seeds={seeds} | stimuli={n} | permutations={a.permutations}")
    print("RDM metric: Pearson correlation distance (1-r)")
    print("RSA: upper triangle Spearman")
    print("Permutation: synchronous stimulus-label row/column shuffle")
    print("z_joint_mean = 0.5*(z_ento+z_mvl), REPRESENTATIONAL READOUT ONLY")
    print("=" * 136)

    for seed in seeds:
        set_seed(seed)
        ckpt = (
            project / "results" / f"seed{seed}"
            / "ento_mvl" / "final_ento_mvl.pth"
        )
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model, meta = build_joint_model(ckpt, device)
        feats = extract(model, loader, device)
        rdms = {k: pearson_distance_rdm(v) for k, v in feats.items()}
        seed_rdms[seed] = rdms

        main_tests = {
            "MVL_to_behavior": stimulus_label_permutation_test(
                rdms["z_mvl"], behavior, a.permutations, seed * 100 + 1
            ),
            "ENTO_to_behavior": stimulus_label_permutation_test(
                rdms["z_ento"], behavior, a.permutations, seed * 100 + 2
            ),
            "JointMean_to_behavior": stimulus_label_permutation_test(
                rdms["z_joint_mean"], behavior, a.permutations, seed * 100 + 3
            ),
            "SourceLogits_to_behavior": stimulus_label_permutation_test(
                rdms["source_logits"], behavior, a.permutations, seed * 100 + 4
            ),
            "FinalLogits_to_behavior": stimulus_label_permutation_test(
                rdms["final_logits"], behavior, a.permutations, seed * 100 + 5
            ),
        }
        qs = bh_fdr([x["p_perm_two_sided"] for x in main_tests.values()])
        for (name, result), q in zip(main_tests.items(), qs):
            per_seed_rows.append({
                "seed": seed,
                "test": name,
                "rho": result["rho"],
                "p_perm_two_sided": result["p_perm_two_sided"],
                "q_bh_within_seed_5_behavior_tests": q,
                "sE": float(meta.get("s_E", 1.0)),
            })

        neural_rows = [
            ("ENTO_to_ENTO_neural", rdms["z_ento"], ento_neural),
            ("ENTO_to_MVL_neural", rdms["z_ento"], mvl_neural),
            ("MVL_to_MVL_neural", rdms["z_mvl"], mvl_neural),
            ("MVL_to_ENTO_neural", rdms["z_mvl"], ento_neural),
            ("JointMean_to_ENTO_neural", rdms["z_joint_mean"], ento_neural),
            ("JointMean_to_MVL_neural", rdms["z_joint_mean"], mvl_neural),
        ]
        for name, model_rdm, target in neural_rows:
            r = stimulus_label_permutation_test(
                model_rdm, target, a.permutations, seed * 1000 + len(per_seed_rows)
            )
            per_seed_rows.append({
                "seed": seed,
                "test": name,
                "rho": r["rho"],
                "p_perm_two_sided": r["p_perm_two_sided"],
                "q_bh_within_seed_5_behavior_tests": "",
                "sE": float(meta.get("s_E", 1.0)),
            })

        delta = stimulus_label_delta_test(
            rdms["z_joint_mean"], rdms["z_mvl"], behavior,
            a.permutations, seed * 100 + 91
        )
        per_seed_rows.append({
            "seed": seed,
            "test": "JointMean_minus_MVL_behavior",
            "rho": delta["delta_rho"],
            "p_perm_two_sided": delta["p_perm_two_sided"],
            "q_bh_within_seed_5_behavior_tests": "",
            "rho_joint": delta["rho_first"],
            "rho_mvl": delta["rho_second"],
            "sE": float(meta.get("s_E", 1.0)),
        })

        np.savez_compressed(
            outdir / f"seed{seed}_model_rdms.npz",
            **rdms,
            stimulus_ids=np.asarray([r["stimulus_id"] for r in rows_manifest]),
        )

        print(
            f"seed={seed} | "
            f"Behavior rho: E={main_tests['ENTO_to_behavior']['rho']:+.4f} "
            f"M={main_tests['MVL_to_behavior']['rho']:+.4f} "
            f"Joint={main_tests['JointMean_to_behavior']['rho']:+.4f} "
            f"FinalLogits={main_tests['FinalLogits_to_behavior']['rho']:+.4f} | "
            f"Joint-MVL={delta['delta_rho']:+.4f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(outdir / "01_per_seed_behavior_neural_RSA.csv", per_seed_rows)

    # Aggregate RDM across pre-specified seeds, then run one formal stimulus-label test.
    repr_keys = ["z_mvl", "z_ento", "z_joint_mean", "source_logits", "final_logits"]
    mean_rdms = {
        k: np.mean(np.stack([seed_rdms[s][k] for s in seeds], axis=0), axis=0)
        for k in repr_keys
    }
    aggregate_rows = []
    for i, (label, key) in enumerate([
        ("MVL_to_behavior", "z_mvl"),
        ("ENTO_to_behavior", "z_ento"),
        ("JointMean_to_behavior", "z_joint_mean"),
        ("SourceLogits_to_behavior", "source_logits"),
        ("FinalLogits_to_behavior", "final_logits"),
    ], start=1):
        r = stimulus_label_permutation_test(
            mean_rdms[key], behavior, a.permutations, 770000 + i
        )
        aggregate_rows.append({
            "test": label,
            "rho_meanRDM": r["rho"],
            "p_perm_two_sided": r["p_perm_two_sided"],
        })
    qs = bh_fdr([r["p_perm_two_sided"] for r in aggregate_rows])
    for r, q in zip(aggregate_rows, qs):
        r["q_bh_5_behavior_tests"] = q

    delta_agg = stimulus_label_delta_test(
        mean_rdms["z_joint_mean"], mean_rdms["z_mvl"], behavior,
        a.permutations, 779991
    )
    aggregate_rows.append({
        "test": "JointMean_minus_MVL_behavior",
        "rho_meanRDM": delta_agg["delta_rho"],
        "p_perm_two_sided": delta_agg["p_perm_two_sided"],
        "q_bh_5_behavior_tests": "",
        "rho_joint": delta_agg["rho_first"],
        "rho_mvl": delta_agg["rho_second"],
    })

    # Aggregate neural alignments.
    for i, (label, key, target) in enumerate([
        ("ENTO_to_ENTO_neural", "z_ento", ento_neural),
        ("MVL_to_MVL_neural", "z_mvl", mvl_neural),
        ("JointMean_to_ENTO_neural", "z_joint_mean", ento_neural),
        ("JointMean_to_MVL_neural", "z_joint_mean", mvl_neural),
    ], start=1):
        r = stimulus_label_permutation_test(
            mean_rdms[key], target, a.permutations, 780000 + i
        )
        aggregate_rows.append({
            "test": label,
            "rho_meanRDM": r["rho"],
            "p_perm_two_sided": r["p_perm_two_sided"],
            "q_bh_5_behavior_tests": "",
        })

    write_csv(outdir / "02_aggregate_meanRDM_RSA.csv", aggregate_rows)
    np.savez_compressed(
        outdir / "aggregate_mean_model_rdms.npz",
        **mean_rdms,
        behavior_rdm=behavior,
        ento_neural_rdm=ento_neural,
        mvl_neural_rdm=mvl_neural,
    )

    # Seed-wise descriptive means/SD.
    seed_summary = []
    for test in sorted({r["test"] for r in per_seed_rows}):
        vals = [
            float(r["rho"]) for r in per_seed_rows
            if r["test"] == test and isinstance(r["rho"], (int, float, np.floating))
        ]
        if vals:
            seed_summary.append({
                "test": test,
                "n_seeds": len(vals),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "positive_seeds": sum(v > 0 for v in vals),
            })
    write_csv(outdir / "03_seed_mean_sd.csv", seed_summary)

    protocol = {
        "architecture": (
            "MVL-like output + calibrated ENTO-like residual across the complete class space; "
            "interaction OFF; extra DeltaM OFF"
        ),
        "evaluation_only": True,
        "behavior_or_neural_used_for_checkpoint_selection": False,
        "rdm_metric": "Pearson correlation distance (1-r)",
        "rsa": "Spearman correlation on upper-triangular RDM entries",
        "permutation": (
            "synchronous stimulus-label permutation: same permutation applied to "
            "RDM rows and columns; pair entries are never independently shuffled"
        ),
        "joint_representation": (
            "z_joint_mean = 0.5*(z_ento+z_mvl); diagnostic representational readout "
            "only, not the classification head"
        ),
        "seeds": seeds,
        "permutations": a.permutations,
        "manifest": str(manifest_path),
        "behavior_rdm": str(behavior_path),
        "ento_neural_rdm": str(ento_path),
        "mvl_neural_rdm": str(mvl_path),
    }
    write_json(outdir / "PROTOCOL.json", protocol)

    print("\n[ALL DONE]")
    print("Per-seed :", outdir / "01_per_seed_behavior_neural_RSA.csv")
    print("Aggregate:", outdir / "02_aggregate_meanRDM_RSA.csv")
    print("Seed mean:", outdir / "03_seed_mean_sd.csv")
    print("Protocol :", outdir / "PROTOCOL.json")


if __name__ == "__main__":
    main()
