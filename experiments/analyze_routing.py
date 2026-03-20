"""
Routing analysis and visualization for AttnRes models.

Generates:
1. Attention weight heatmaps (depth x depth)
2. Per-block importance bar charts
3. Input-dependent routing comparisons (easy vs hard)
4. Accuracy vs FLOPs Pareto curves
5. Routing entropy over training

Usage:
    python experiments/analyze_routing.py --checkpoint outputs/standard/best.pt
    python experiments/analyze_routing.py --checkpoint outputs/standard/best.pt --sweep_results outputs/standard/final_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive_transformer import (
    AdaptiveTransformerConfig,
    AdaptiveTransformerForCausalLM,
)
from src.data import StructuredSyntheticLM


def collect_routing_statistics(model, dataloader, device, num_batches=50):
    """Collect routing weight statistics across a calibration set."""
    model.eval()
    all_matrices = []
    all_block_importance = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            stats = model.model.get_routing_statistics(input_ids)
            all_matrices.append(stats["importance_matrix"])
            all_block_importance.append(stats["block_importance"])

    return all_matrices, all_block_importance


def collect_per_difficulty_stats(model, vocab_size, seq_len, device, n_samples=500):
    """Collect routing stats separately for easy/medium/hard inputs."""
    model.eval()
    results = {}

    for diff in ["easy", "mixed", "hard"]:
        ds = StructuredSyntheticLM(
            vocab_size=vocab_size, seq_len=seq_len,
            num_samples=n_samples, difficulty_mix=diff, seed=42,
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        matrices, importance = collect_routing_statistics(model, loader, device, num_batches=16)
        results[diff] = {
            "matrices": matrices,
            "importance": importance,
            "avg_matrix": torch.stack(matrices).mean(0).numpy(),
        }

    return results


def plot_attention_heatmap(importance_matrices, save_path, title="AttnRes Weight Matrix"):
    """Plot average attention weight heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg_matrix = torch.stack(importance_matrices).mean(0).numpy()
    N = avg_matrix.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(avg_matrix, cmap="Blues", aspect="auto", vmin=0)

    ax.set_xlabel("Target Block (l)", fontsize=13)
    ax.set_ylabel("Source Block (i)", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f"B{i}" for i in range(N)])
    ax.set_yticklabels([f"B{i}" for i in range(N)])

    for i in range(N):
        for j in range(N):
            val = avg_matrix[i, j]
            if val > 0.005:
                color = "white" if val > 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       color=color, fontsize=9, fontweight="bold" if val > 0.2 else "normal")

    plt.colorbar(im, ax=ax, label="Attention Weight (α)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_block_importance(all_importance, save_path, thresholds=None):
    """Plot per-block importance scores with skip threshold lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Convert to numpy
    all_imp = np.array(all_importance)
    mean_imp = all_imp.mean(axis=0)
    std_imp = all_imp.std(axis=0)
    N = len(mean_imp)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bars = ax.bar(range(N), mean_imp, yerr=std_imp, capsize=4,
                  alpha=0.8, color="steelblue", edgecolor="navy", linewidth=0.5)

    if thresholds is None:
        thresholds = {"eps=0.01": 0.01, "eps=0.05": 0.05}

    colors = ["red", "orange", "green"]
    for (label, val), color in zip(thresholds.items(), colors):
        ax.axhline(y=val, color=color, linestyle="--", linewidth=1.5, label=label)
        # Count skippable blocks
        n_skip = sum(1 for x in mean_imp[1:-1] if x < val)  # Never skip first/last
        ax.text(N - 0.5, val + 0.002, f"skip {n_skip} blocks", fontsize=8,
               color=color, ha="right")

    ax.set_xlabel("Block Index", fontsize=13)
    ax.set_ylabel("Importance Score I(n)", fontsize=13)
    ax.set_title("Per-Block Importance: max downstream attention weight", fontsize=14)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"B{i}" for i in range(N)])
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_difficulty_comparison(per_diff_stats, save_path):
    """Compare routing patterns across input difficulties."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (diff, stats) in enumerate(per_diff_stats.items()):
        ax = axes[idx]
        matrix = stats["avg_matrix"]
        N = matrix.shape[0]

        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"{diff.capitalize()} Inputs", fontsize=14)
        ax.set_xlabel("Target Block")
        ax.set_ylabel("Source Block")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))

        for i in range(N):
            for j in range(N):
                val = matrix[i, j]
                if val > 0.01:
                    color = "white" if val > 0.4 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           color=color, fontsize=8)

    plt.colorbar(im, ax=axes, label="α weight", shrink=0.8)
    plt.suptitle("Routing Patterns by Input Difficulty", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pareto_curve(sweep_results, save_path):
    """Plot accuracy vs FLOPs Pareto curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Extract data
    thresholds = [r["threshold"] for r in sweep_results]
    ppls = [r["perplexity"] for r in sweep_results]
    flops = [r["flops_ratio"] for r in sweep_results]
    blocks = [r["avg_blocks"] for r in sweep_results]

    # PPL vs FLOPs
    ax1.plot(flops, ppls, "o-", color="steelblue", markersize=8, linewidth=2)
    for i, t in enumerate(thresholds):
        ax1.annotate(f"ε={t}", (flops[i], ppls[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    # Highlight Pareto optimal points
    baseline_ppl = ppls[0]
    for i, (f, p) in enumerate(zip(flops, ppls)):
        if p <= baseline_ppl * 1.02:  # Within 2% of baseline
            ax1.plot(f, p, "s", color="red", markersize=12, zorder=5, alpha=0.5)

    ax1.set_xlabel("Effective FLOPs Ratio", fontsize=13)
    ax1.set_ylabel("Perplexity", fontsize=13)
    ax1.set_title("Perplexity vs. Compute (Pareto Curve)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Blocks vs threshold
    ax2.plot(thresholds, blocks, "s-", color="coral", markersize=8, linewidth=2)
    total = sweep_results[0]["total_blocks"]
    ax2.axhline(y=total, color="gray", linestyle=":", label=f"Full depth ({total} blocks)")
    ax2.set_xlabel("Skip Threshold (ε)", fontsize=13)
    ax2.set_ylabel("Average Blocks Executed", fontsize=13)
    ax2.set_title("Effective Depth vs. Threshold", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(log_path, save_path):
    """Plot training loss and routing entropy curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = []
    with open(log_path) as f:
        for line in f:
            entries.append(json.loads(line))

    # Separate step-level and epoch-level entries
    step_entries = [e for e in entries if "step" in e and "train_loss" in e]
    epoch_entries = [e for e in entries if "epoch" in e and "val_loss" in e]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training loss
    if step_entries:
        steps = [e["step"] for e in step_entries]
        losses = [e["train_loss"] for e in step_entries]
        axes[0].plot(steps, losses, alpha=0.5, linewidth=0.5, color="blue")
        # Smoothed
        window = min(20, len(losses) // 5 + 1)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            axes[0].plot(steps[window-1:], smoothed, linewidth=2, color="darkblue")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # Validation loss/PPL
    if epoch_entries:
        epochs = [e["epoch"] for e in epoch_entries]
        val_losses = [e["val_loss"] for e in epoch_entries]
        val_ppls = [e.get("val_ppl", math.exp(min(l, 20))) for l, e in zip(val_losses, epoch_entries)]
        axes[1].plot(epochs, val_ppls, "o-", color="green", markersize=6, linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Perplexity")
    axes[1].set_title("Validation Perplexity")
    axes[1].grid(True, alpha=0.3)

    # Routing entropy
    entropy_entries = [e for e in step_entries if "routing_entropy" in e and e["routing_entropy"] != 0]
    if entropy_entries:
        steps = [e["step"] for e in entropy_entries]
        entropies = [e["routing_entropy"] for e in entropy_entries]
        axes[2].plot(steps, entropies, alpha=0.5, linewidth=0.5, color="purple")
        if len(entropies) > 5:
            window = min(20, len(entropies) // 5 + 1)
            smoothed = np.convolve(entropies, np.ones(window)/window, mode="valid")
            axes[2].plot(steps[window-1:], smoothed, linewidth=2, color="darkviolet")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Routing Entropy")
    axes[2].set_title("AttnRes Routing Entropy")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


import math


def main():
    parser = argparse.ArgumentParser(description="Analyze AttnRes routing patterns")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sweep_results", type=str, default="", help="Path to final_results.json for Pareto plot")
    parser.add_argument("--train_log", type=str, default="", help="Path to train_log.jsonl for training curves")
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint), "analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt["config"]
    model = AdaptiveTransformerForCausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    print(f"Model: d={config.d_model}, L={config.n_layers}, N={config.n_blocks}")
    print(f"Val loss: {ckpt.get('val_loss', '?')}, PPL: {ckpt.get('val_ppl', '?')}")

    # Collect routing statistics
    print("\nCollecting routing statistics...")
    ds = StructuredSyntheticLM(
        vocab_size=config.vocab_size, seq_len=min(256, config.max_seq_len),
        num_samples=2000, difficulty_mix="mixed", seed=42,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    matrices, importance_scores = collect_routing_statistics(model, loader, args.device)

    # Generate plots
    print("\nGenerating visualizations...")

    # 1. Attention heatmap
    plot_attention_heatmap(
        matrices,
        os.path.join(args.output_dir, "routing_heatmap.png"),
        title=f"AttnRes Weight Matrix (d={config.d_model}, L={config.n_layers})",
    )

    # 2. Block importance
    plot_block_importance(
        importance_scores,
        os.path.join(args.output_dir, "block_importance.png"),
    )

    # 3. Per-difficulty comparison
    print("  Analyzing per-difficulty routing...")
    per_diff = collect_per_difficulty_stats(
        model, config.vocab_size, min(256, config.max_seq_len), args.device,
    )
    plot_difficulty_comparison(
        per_diff,
        os.path.join(args.output_dir, "difficulty_comparison.png"),
    )

    # 4. Pareto curve from sweep results
    if args.sweep_results and os.path.exists(args.sweep_results):
        with open(args.sweep_results) as f:
            results_data = json.load(f)
        sweep = results_data.get("skip_sweep", [])
        if sweep:
            plot_pareto_curve(sweep, os.path.join(args.output_dir, "pareto_curve.png"))

    # 5. Training curves
    if args.train_log and os.path.exists(args.train_log):
        plot_training_curves(args.train_log, os.path.join(args.output_dir, "training_curves.png"))
    else:
        # Auto-detect
        log_dir = os.path.dirname(args.checkpoint)
        log_path = os.path.join(log_dir, "train_log.jsonl")
        if os.path.exists(log_path):
            plot_training_curves(log_path, os.path.join(args.output_dir, "training_curves.png"))

    # Print summary statistics
    print(f"\n{'='*50}")
    print("Routing Summary")
    print(f"{'='*50}")
    avg_imp = np.array(importance_scores).mean(axis=0)
    for i, imp in enumerate(avg_imp):
        skippable = "SKIP" if imp < 0.01 and 0 < i < len(avg_imp) - 1 else "KEEP"
        print(f"  Block {i}: importance={imp:.4f} [{skippable}]")

    # Per-difficulty depth analysis
    print(f"\nEffective depth by difficulty (eps=0.01):")
    for diff, stats in per_diff.items():
        avg_imp_diff = np.array(stats["importance"]).mean(axis=0)
        kept = sum(1 for i, x in enumerate(avg_imp_diff) if x >= 0.01 or i == 0 or i == len(avg_imp_diff) - 1)
        print(f"  {diff}: {kept}/{len(avg_imp_diff)} blocks")

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
