#!/usr/bin/env python3
"""Plot heldout math evaluation results as grouped bar charts.

Usage:
    python plot_heldout_bar.py --results-root /home/fengdic/scratch/hint_rl_results/logs/fengdic
    python plot_heldout_bar.py --results-root /path/to/logs --k-values 1 8 32 --output bar.png
"""

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ── Configuration ────────────────────────────────────────────────────────────
DATASETS = ["aime24", "aime25", "olympiad_bench", "hmmt_feb_2025"]
DATASET_LABELS = {
    "aime24": "AIME 2024",
    "aime25": "AIME 2025",
    "olympiad_bench": "OlympiadBench",
    "hmmt_feb_2025": "HMMT Feb 2025",
}

# model_tag → (display_label, color)
# Add new models here as they become available.
MODEL_REGISTRY = {
    "hintrl": ("Hint RL (Nemotron 1.5B)", "#55A868"),
    "questa": ("QuestA (Nemotron 1.5B)", "#C44E52"),
    "nemotron_base": ("OpenMath-Nemotron-1.5B", "#4C72B0"),
    "deepscaler_base": ("DeepScaleR-1.5B", "#DD8452"),
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_pass_at_k(rewards, k):
    """Compute Pass@K for a single problem.

    Uses the unbiased estimator: Pass@K = 1 - C(n-c, k) / C(n, k).
    For k=1 this simplifies to c/n.
    """
    n = len(rewards)
    c = sum(r > 0 for r in rewards)
    if k <= 0 or k > n:
        return c / n  # fallback to simple average
    if k == 1:
        return c / n
    if n - c < k:
        return 1.0
    from math import comb

    return 1.0 - comb(n - c, k) / comb(n, k)


def load_results(results_root, dataset_name, trial_name):
    """Load all JSONL result files for one (model, dataset) eval run.

    Returns list of per-problem reward lists, or None if directory missing.
    """
    rollout_dir = os.path.join(
        results_root, f"eval_{dataset_name}", trial_name, "rollout", "0"
    )
    if not os.path.isdir(rollout_dir):
        return None

    all_rewards = []
    for filename in sorted(os.listdir(rollout_dir), key=lambda f: int(f.split(".")[0]) if f.split(".")[0].isdigit() else 0):
        if not filename.endswith(".jsonl"):
            continue
        filepath = os.path.join(rollout_dir, filename)
        rewards = []
        with open(filepath) as f:
            for line in f:
                obj = json.loads(line)
                rewards.append(obj.get("reward", 0.0))
        if rewards:
            all_rewards.append(rewards)

    return all_rewards if all_rewards else None


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Plot heldout math eval results")
    parser.add_argument(
        "--results-root",
        type=str,
        default="/home/fengdic/scratch/hint_rl_results/logs/fengdic",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help="Model tags to include (keys in MODEL_REGISTRY)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="K values for Pass@K",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="heldout_eval_bar.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models = {tag: MODEL_REGISTRY[tag] for tag in args.models if tag in MODEL_REGISTRY}

    if not models:
        print(f"No valid model tags. Available: {list(MODEL_REGISTRY.keys())}")
        return

    # ── Collect results ──────────────────────────────────────────────────
    # results[k][model_tag][dataset] = pass_at_k_value or None
    results = {}
    for k in args.k_values:
        results[k] = {}
        for model_tag in models:
            results[k][model_tag] = {}
            for ds in DATASETS:
                trial_name = f"eval_{ds}-{model_tag}"
                rewards_per_problem = load_results(args.results_root, ds, trial_name)
                if rewards_per_problem is None:
                    results[k][model_tag][ds] = None
                    continue
                pass_k_values = [compute_pass_at_k(r, k) for r in rewards_per_problem]
                results[k][model_tag][ds] = np.mean(pass_k_values)

    # ── Print summary table ──────────────────────────────────────────────
    for k in args.k_values:
        print(f"\n{'='*60}")
        print(f"  Pass@{k}")
        print(f"{'='*60}")
        header = f"{'Dataset':<20}" + "".join(
            f"{label:<25}" for label, _ in models.values()
        )
        print(header)
        print("-" * len(header))
        for ds in DATASETS:
            row = f"{DATASET_LABELS[ds]:<20}"
            for model_tag in models:
                val = results[k][model_tag][ds]
                if val is not None:
                    row += f"{val:<25.4f}"
                else:
                    row += f"{'(no data)':<25}"
            print(row)

    # ── Check if we have any data to plot ────────────────────────────────
    has_data = any(
        results[k][m][ds] is not None
        for k in args.k_values
        for m in models
        for ds in DATASETS
    )
    if not has_data:
        print("\nNo evaluation results found yet. Run evals first, then re-run this script.")
        return

    # ── Plot ─────────────────────────────────────────────────────────────
    n_k = len(args.k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 5), layout="constrained")
    if n_k == 1:
        axes = [axes]

    x = np.arange(len(DATASETS))
    width = 0.8 / len(models)

    for ax_idx, k in enumerate(args.k_values):
        ax = axes[ax_idx]
        for m_idx, (model_tag, (model_label, color)) in enumerate(models.items()):
            values = []
            for ds in DATASETS:
                val = results[k][model_tag].get(ds)
                values.append(val if val is not None else 0)

            offset = (m_idx - len(models) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, values, width, label=model_label, color=color
            )

            # Value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.1%}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_xlabel("Dataset")
        ax.set_ylabel(f"Pass@{k}")
        ax.set_title(f"Pass@{k}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [DATASET_LABELS[ds] for ds in DATASETS], rotation=20, ha="right"
        )
        all_vals = [
            results[k][m].get(ds) or 0
            for m in models
            for ds in DATASETS
        ]
        ymax = min(1.0, max(all_vals) * 1.15 + 0.02)
        ymin = max(0.0, min(v for v in all_vals if v > 0) * 0.88)
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=8)

    fig.suptitle("Heldout Math Evaluation — Base Models", fontsize=14)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
