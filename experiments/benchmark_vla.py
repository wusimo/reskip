"""
VLA benchmark: Train and evaluate modality-adaptive depth.

Compares:
1. AttnRes VLA (full depth, learned routing)
2. AttnRes VLA + uniform skipping
3. AttnRes VLA + modality-aware skipping

Measures:
- Action prediction accuracy (L1 error) by task type
- Per-modality effective depth
- Inference throughput
- Routing pattern analysis

Usage:
    python experiments/benchmark_vla.py
    python experiments/benchmark_vla.py --task_type pick_place --epochs 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vla_adaptive import VLAAdaptiveConfig, VLAAdaptiveTransformer, TokenModality
from src.data import create_vla_dataloaders, StructuredVLADataset
from src.utils import TrainLogger, save_results, profile_latency, CosineWarmupScheduler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_vla(config, train_loader, val_loader, args):
    """Train VLA model with validation."""
    model = VLAAdaptiveTransformer(config)
    device = torch.device(args.device)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  VLA parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineWarmupScheduler(optimizer, min(100, total_steps // 10), total_steps)

    best_val_l1 = float("inf")
    logger = TrainLogger(args.output_dir, "vla_train")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            vision = batch["vision_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            actions = batch["target_actions"].to(device)

            result = model(
                input_ids=input_ids,
                vision_features=vision,
                target_actions=actions,
            )

            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch * len(train_loader) + n_batches)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Validate
        val_l1 = evaluate_vla_actions(model, val_loader, device)

        logger.log(
            {"epoch": epoch + 1, "train_loss": avg_loss, "val_l1": val_l1},
            f"  Epoch {epoch+1}/{args.epochs}: train_loss={avg_loss:.4f} val_l1={val_l1:.4f}"
        )

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_l1": val_l1,
            }, os.path.join(args.output_dir, "vla_best.pt"))

    logger.close()
    return model, best_val_l1


@torch.no_grad()
def evaluate_vla_actions(model, val_loader, device):
    """Evaluate action prediction L1 error."""
    model.eval()
    total_l1 = 0.0
    total_samples = 0

    for batch in val_loader:
        vision = batch["vision_features"].to(device)
        input_ids = batch["input_ids"].to(device)
        actions = batch["target_actions"].to(device)

        result = model(input_ids=input_ids, vision_features=vision)
        pred = result["actions"]

        # Truncate/pad if shapes differ
        min_chunk = min(pred.shape[1], actions.shape[1])
        l1 = (pred[:, :min_chunk] - actions[:, :min_chunk]).abs().mean().item()
        total_l1 += l1 * input_ids.shape[0]
        total_samples += input_ids.shape[0]

    return total_l1 / max(total_samples, 1)


@torch.no_grad()
def evaluate_vla_with_routing(model, val_loader, device, skip_mode="none"):
    """Evaluate VLA with routing analysis."""
    model.eval()

    if skip_mode == "modality_aware":
        model.config.enable_skipping = True
        model.config.modality_aware_routing = True
    elif skip_mode == "uniform":
        model.config.enable_skipping = True
        model.config.modality_aware_routing = False
    else:
        model.config.enable_skipping = False

    total_l1 = 0.0
    total_samples = 0
    per_task_l1 = {}
    modality_depths = {mod.name: [] for mod in TokenModality}
    blocks_executed_list = []

    start_time = time.time()

    for batch in val_loader:
        vision = batch["vision_features"].to(device)
        input_ids = batch["input_ids"].to(device)
        actions = batch["target_actions"].to(device)
        task_types = batch.get("task_type", [])

        result = model(
            input_ids=input_ids,
            vision_features=vision,
            return_routing_info=True,
        )

        pred = result["actions"]
        min_chunk = min(pred.shape[1], actions.shape[1])
        batch_l1 = (pred[:, :min_chunk] - actions[:, :min_chunk]).abs()

        # Per-sample L1
        for j in range(input_ids.shape[0]):
            sample_l1 = batch_l1[j].mean().item()
            total_l1 += sample_l1
            total_samples += 1

            if j < len(task_types):
                task = task_types[j]
                if task not in per_task_l1:
                    per_task_l1[task] = []
                per_task_l1[task].append(sample_l1)

        # Blocks executed
        if "blocks_executed" in result:
            n_exec = sum(1 for _, s in result["blocks_executed"] if s >= 0)
            blocks_executed_list.append(n_exec)

        # Per-modality depth from routing weights
        if "per_modality_weights" in result:
            for mod in TokenModality:
                mod_weights = result["per_modality_weights"][mod]
                if mod_weights and len(mod_weights) > 0:
                    # Each entry is a tensor of different shape depending on block
                    # Compute effective depth as number of blocks with significant weight
                    depths = []
                    for w in mod_weights:
                        if w.dim() == 1:
                            # (num_sources,) - weight on last source indicates importance
                            depths.append((w > 0.01).sum().item())
                        elif w.dim() == 0:
                            depths.append(1 if w.item() > 0.01 else 0)
                    if depths:
                        modality_depths[mod.name].append(np.mean(depths))

    elapsed = time.time() - start_time
    avg_l1 = total_l1 / max(total_samples, 1)

    results = {
        "skip_mode": skip_mode,
        "avg_l1_error": avg_l1,
        "throughput_samples_per_sec": total_samples / max(elapsed, 0.001),
        "total_time_sec": elapsed,
        "per_task_l1": {k: np.mean(v) for k, v in per_task_l1.items()},
    }

    if blocks_executed_list:
        results["avg_blocks_executed"] = np.mean(blocks_executed_list)
        results["total_blocks"] = model.config.n_blocks

    for mod_name, depths in modality_depths.items():
        if depths:
            results[f"{mod_name}_avg_depth"] = np.mean(depths)

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_vla_results(all_results, output_dir):
    """Generate VLA benchmark visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. L1 error by skip mode
    modes = [r["skip_mode"] for r in all_results]
    l1s = [r["avg_l1_error"] for r in all_results]
    colors = ["steelblue", "coral", "green"]
    axes[0].bar(modes, l1s, color=colors[:len(modes)], alpha=0.8, edgecolor="black")
    axes[0].set_ylabel("L1 Error", fontsize=13)
    axes[0].set_title("Action Prediction Error by Skip Mode", fontsize=14)
    for i, v in enumerate(l1s):
        axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=10)

    # 2. Per-task L1
    task_data = {}
    for r in all_results:
        mode = r["skip_mode"]
        for task, l1 in r.get("per_task_l1", {}).items():
            if task not in task_data:
                task_data[task] = {}
            task_data[task][mode] = l1

    if task_data:
        tasks = list(task_data.keys())
        x = np.arange(len(tasks))
        width = 0.25
        for i, mode in enumerate(modes):
            vals = [task_data.get(t, {}).get(mode, 0) for t in tasks]
            axes[1].bar(x + i * width, vals, width, label=mode, alpha=0.8)
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(tasks, rotation=15)
        axes[1].set_ylabel("L1 Error")
        axes[1].set_title("Per-Task L1 Error", fontsize=14)
        axes[1].legend()

    # 3. Modality depth
    modality_data = {}
    for r in all_results:
        if r["skip_mode"] == "none":  # Full depth analysis
            for mod in TokenModality:
                key = f"{mod.name}_avg_depth"
                if key in r:
                    modality_data[mod.name] = r[key]

    if modality_data:
        mods = list(modality_data.keys())
        depths = list(modality_data.values())
        bar_colors = ["skyblue", "lightgreen", "salmon"]
        axes[2].bar(mods, depths, color=bar_colors[:len(mods)], alpha=0.8, edgecolor="black")
        axes[2].set_ylabel("Average Effective Depth")
        axes[2].set_title("Per-Modality Effective Depth", fontsize=14)
        for i, v in enumerate(depths):
            axes[2].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=11)

    plt.suptitle("VLA AttnRes Benchmark Results", fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "vla_benchmark.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_modality_routing_heatmap(model, val_loader, device, output_dir):
    """Analyze and plot per-modality routing patterns."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    model.config.enable_skipping = False

    # Get one batch for analysis
    batch = next(iter(val_loader))
    vision = batch["vision_features"].to(device)
    input_ids = batch["input_ids"].to(device)

    with torch.no_grad():
        analysis = model.analyze_modality_depth(input_ids, vision)

    if not analysis:
        return

    fig, axes = plt.subplots(1, len(analysis), figsize=(6 * len(analysis), 5))
    if len(analysis) == 1:
        axes = [axes]

    for idx, (mod_name, stats) in enumerate(analysis.items()):
        ax = axes[idx]
        weights = stats["mean_weights_per_block"].numpy()

        if weights.ndim == 2:
            im = ax.imshow(weights, cmap="YlOrRd", aspect="auto")
            ax.set_xlabel("Source Block")
            ax.set_ylabel("Depth Position")
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.bar(range(len(weights)), weights.flatten(), color="steelblue")
            ax.set_xlabel("Source")
            ax.set_ylabel("Weight")

        ax.set_title(f"{mod_name}\n(eff. depth={stats['effective_depth']}, conc={stats['concentration']:.3f})")

    plt.suptitle("Per-Modality AttnRes Routing", fontsize=15)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "modality_routing.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(args):
    """Run full VLA benchmark."""
    os.makedirs(args.output_dir, exist_ok=True)

    config = VLAAdaptiveConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_blocks=args.n_blocks,
        d_ff=args.d_model * 4,
        vocab_size=args.vocab_size,
        max_seq_len=512,
        vision_dim=args.vision_dim,
        vision_seq_len=args.vision_seq_len,
        action_dim=args.action_dim,
        action_chunk_size=args.action_chunk,
        enable_skipping=False,
        modality_aware_routing=True,
    )

    vla_kwargs = dict(
        vision_dim=args.vision_dim,
        vision_seq_len=args.vision_seq_len,
        lang_seq_len=args.seq_len,
        action_dim=args.action_dim,
        action_chunk_size=args.action_chunk,
        vocab_size=args.vocab_size,
    )

    train_loader, val_loader = create_vla_dataloaders(
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        task_type=args.task_type,
        **vla_kwargs,
    )

    print(f"{'='*60}")
    print(f"VLA Benchmark: AttnRes Modality-Adaptive Depth")
    print(f"{'='*60}")
    print(f"Tasks: {args.task_type}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    print(f"Backbone: d={config.d_model}, L={config.n_layers}, N={config.n_blocks}")
    print(f"Vision: dim={config.vision_dim}, seq={config.vision_seq_len}")
    print(f"Action: dim={config.action_dim}, chunk={config.action_chunk_size}")
    print(f"{'='*60}")

    # Train
    print("\nTraining VLA model...")
    model, best_l1 = train_vla(config, train_loader, val_loader, args)
    print(f"Best validation L1: {best_l1:.4f}")

    # Evaluate with different skip modes
    print(f"\n{'='*60}")
    print("Evaluating skip modes...")
    print(f"{'='*60}")

    all_results = []
    for mode in ["none", "uniform", "modality_aware"]:
        print(f"\n  Mode: {mode}")
        results = evaluate_vla_with_routing(model, val_loader, torch.device(args.device), skip_mode=mode)
        all_results.append(results)

        print(f"    L1 error: {results['avg_l1_error']:.4f}")
        print(f"    Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
        if "avg_blocks_executed" in results:
            print(f"    Blocks: {results['avg_blocks_executed']:.1f}/{results.get('total_blocks', '?')}")
        for key, val in sorted(results.items()):
            if "depth" in key:
                print(f"    {key}: {val:.2f}")
        if results.get("per_task_l1"):
            for task, l1 in results["per_task_l1"].items():
                print(f"    {task}: L1={l1:.4f}")

    # Latency comparison
    print(f"\n{'='*60}")
    print("Latency Profiling")
    print(f"{'='*60}")

    sample_vision = torch.randn(1, config.vision_seq_len, config.vision_dim, device=args.device)
    sample_ids = torch.randint(0, config.vocab_size, (1, args.seq_len), device=args.device)

    # Full depth
    model.config.enable_skipping = False
    lat_full = profile_latency(model, sample_ids, vision_features=sample_vision)
    print(f"  Full depth: {lat_full['mean_ms']:.1f}ms")

    # With skipping
    model.config.enable_skipping = True
    model.config.modality_aware_routing = True
    lat_skip = profile_latency(model, sample_ids, vision_features=sample_vision)
    speedup = lat_full["mean_ms"] / max(lat_skip["mean_ms"], 0.01)
    print(f"  Modality-aware skip: {lat_skip['mean_ms']:.1f}ms ({speedup:.2f}x)")

    # Modality depth analysis
    print(f"\n{'='*60}")
    print("Per-Modality Depth Analysis")
    print(f"{'='*60}")
    model.config.enable_skipping = False

    with torch.no_grad():
        batch = next(iter(val_loader))
        analysis = model.analyze_modality_depth(
            batch["input_ids"].to(args.device),
            batch["vision_features"].to(args.device),
        )
        for mod_name, stats in analysis.items():
            print(f"  {mod_name}:")
            print(f"    Effective depth: {stats['effective_depth']}")
            print(f"    Weight concentration: {stats['concentration']:.3f}")

    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")
    plot_vla_results(all_results, args.output_dir)
    plot_modality_routing_heatmap(model, val_loader, torch.device(args.device), args.output_dir)

    # Save results
    final = {
        "config": {k: v for k, v in vars(config).items()},
        "best_val_l1": best_l1,
        "skip_mode_results": all_results,
        "latency_full": lat_full,
        "latency_skip": lat_skip,
        "speedup": speedup,
        "modality_analysis": {
            k: {"effective_depth": v["effective_depth"], "concentration": v["concentration"]}
            for k, v in analysis.items()
        },
    }
    save_results(final, args.output_dir, "vla_results.json")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Best val L1: {best_l1:.4f}")
    print(f"Full depth latency: {lat_full['mean_ms']:.1f}ms")
    print(f"Skip latency: {lat_skip['mean_ms']:.1f}ms ({speedup:.2f}x speedup)")
    for mode_result in all_results:
        print(f"  {mode_result['skip_mode']}: L1={mode_result['avg_l1_error']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="VLA AttnRes Benchmark")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--vision_dim", type=int, default=512)
    parser.add_argument("--vision_seq_len", type=int, default=32)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_train", type=int, default=3000)
    parser.add_argument("--num_val", type=int, default=500)
    parser.add_argument("--task_type", type=str, default="mixed",
                       choices=["reach", "push", "pick_place", "mixed"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs/vla")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
