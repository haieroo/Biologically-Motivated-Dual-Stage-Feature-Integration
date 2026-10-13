# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--profile", choices=["quick", "core"], default="core")
    return p.parse_args()


def main():
    a = parse_args()
    root = Path(a.project_root).resolve()
    beh = root / "results" / "final_behavior_rsa_3seeds" / "01_per_seed_behavior_neural_RSA.csv"
    rob = root / "results" / f"final_robustness_3seeds_{a.profile}" / "01_per_seed_condition_results.csv"
    out = root / "results" / "final_posthoc_figures"
    out.mkdir(parents=True, exist_ok=True)

    if beh.exists():
        d = pd.read_csv(beh)
        order = ["ENTO_to_behavior", "MVL_to_behavior", "JointMean_to_behavior"]
        dd = d[d["test"].isin(order)].copy()
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        xs = np.arange(len(order))
        means = [dd.loc[dd["test"] == k, "rho"].mean() for k in order]
        sds = [dd.loc[dd["test"] == k, "rho"].std(ddof=1) for k in order]
        ax.bar(xs, means, yerr=sds, capsize=4)
        for i, k in enumerate(order):
            vals = dd.loc[dd["test"] == k, "rho"].to_numpy()
            ax.scatter(np.full(len(vals), i), vals, zorder=3)
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["ENTO-like", "MVL-like", "Joint mean"])
        ax.set_ylabel("Spearman ρ with behavior RDM")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "behavior_RSA_3seeds.pdf", bbox_inches="tight")
        fig.savefig(out / "behavior_RSA_3seeds.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    if rob.exists():
        d = pd.read_csv(rob)
        d = d[d["family"] != "clean"].copy()
        families = ["occlusion", "patch_shuffle", "gaussian_blur", "gaussian_noise", "color_shift"]
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for family in families:
            sub = d[d["family"] == family]
            if sub.empty:
                continue
            g = sub.groupby("severity", as_index=False)["delta_top1_pp"].mean()
            ax.plot(g["severity"], g["delta_top1_pp"], marker="o", label=family)
        ax.axhline(0, linewidth=0.8)
        ax.set_xlabel("Perturbation severity")
        ax.set_ylabel("Final − MVL-main Top-1 (pp)")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"classification_robustness_delta_{a.profile}.pdf", bbox_inches="tight")
        fig.savefig(out / f"classification_robustness_delta_{a.profile}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for family in families:
            sub = d[d["family"] == family]
            if sub.empty:
                continue
            g = sub.groupby("severity", as_index=False)["joint_retention"].mean()
            ax.plot(g["severity"], g["joint_retention"], marker="o", label=family)
        ax.set_xlabel("Perturbation severity")
        ax.set_ylabel("Joint representational retention (Spearman ρ)")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"joint_RDM_retention_{a.profile}.pdf", bbox_inches="tight")
        fig.savefig(out / f"joint_RDM_retention_{a.profile}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    print("Figures:", out)


if __name__ == "__main__":
    main()
