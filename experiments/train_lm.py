"""
Training script for AttnRes language models with adaptive computation.

Supports:
1. Standard AttnRes: Train with Block AttnRes, evaluate with skip sweeps
2. Looping AttnRes: Train with weight-shared blocks + adaptive halting
3. Baseline: Standard transformer without AttnRes (for comparison)

Usage:
    # Standard AttnRes on structured synthetic data
    python experiments/train_lm.py --mode standard --dataset structured_synthetic

    # Standard AttnRes on WikiText-2
    python experiments/train_lm.py --mode standard --dataset wikitext2

    # Looping AttnRes
    python experiments/train_lm.py --mode looping --n_unique_blocks 4 --max_loops 3

    # Evaluation with skip threshold sweep
    python experiments/train_lm.py --mode eval --checkpoint outputs/standard/best.pt
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive_transformer import (
    AdaptiveTransformerConfig,
    AdaptiveTransformerForCausalLM,
)
from src.looping_transformer import (
    LoopingTransformerConfig,
    LoopingTransformerWithAttnRes,
)
from src.data import create_lm_dataloaders, StructuredSyntheticLM
from src.utils import (
    count_transformer_flops,
    count_effective_flops,
    profile_latency,
    TrainLogger,
    save_results,
    load_yaml_config,
    CosineWarmupScheduler,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_standard(args):
    """Train standard AttnRes model with proper evaluation."""
    # Data
    train_loader, val_loader, actual_vocab = create_lm_dataloaders(
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_train=args.num_train,
        num_val=args.num_val,
    )

    config = AdaptiveTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_blocks=args.n_blocks,
        d_ff=args.d_model * 4,
        vocab_size=actual_vocab,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        attn_res_temperature=args.temperature,
        skip_threshold=args.skip_threshold,
        enable_skipping=False,
    )

    model = AdaptiveTransformerForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    device = torch.device(args.device)
    model = model.to(device)

    # FLOP estimate
    flops = count_transformer_flops(
        config.d_model, config.n_heads, config.d_ff,
        args.seq_len, config.n_layers, actual_vocab, args.batch_size,
    )

    print(f"{'='*60}")
    print(f"Training Standard AttnRes Model")
    print(f"{'='*60}")
    print(f"Parameters: {n_params:,}")
    print(f"FLOPs/batch: {flops['total_gflops']:.2f} GFLOPs")
    print(f"Dataset: {args.dataset} (train={len(train_loader.dataset)}, val={len(val_loader.dataset)})")
    print(f"Vocab: {actual_vocab}, Seq len: {args.seq_len}")
    print(f"Blocks: {config.n_blocks}, Layers/block: {config.n_layers // config.n_blocks}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_steps, total_steps)

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = TrainLogger(args.output_dir, "train")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_entropy = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(input_ids, labels=labels, return_routing_info=True)
                loss = result["loss"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(global_step)

            epoch_loss += loss.item()
            entropy = result.get("routing_entropy", 0)
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.item()
            epoch_entropy += entropy
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                logger.log(
                    {"step": global_step, "train_loss": loss.item(),
                     "lr": optimizer.param_groups[0]["lr"],
                     "routing_entropy": entropy, "epoch": epoch},
                    f"  step {global_step}: loss={loss.item():.4f} entropy={entropy:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_train_loss = epoch_loss / n_batches
        avg_entropy = epoch_entropy / n_batches

        # --- Validate ---
        val_loss, val_ppl = evaluate_model(model, val_loader, device)

        logger.log(
            {"epoch": epoch + 1, "train_loss": avg_train_loss,
             "val_loss": val_loss, "val_ppl": val_ppl,
             "routing_entropy": avg_entropy},
            f"Epoch {epoch+1}/{args.epochs}: train={avg_train_loss:.4f} val={val_loss:.4f} ppl={val_ppl:.2f} entropy={avg_entropy:.4f}"
        )

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            print(f"  -> New best model (val_loss={val_loss:.4f})")

    logger.close()

    # --- Post-training analysis ---
    print(f"\n{'='*60}")
    print("Post-training Analysis")
    print(f"{'='*60}")

    # Latency profiling
    sample_input = torch.randint(0, actual_vocab, (1, args.seq_len), device=device)
    latency = profile_latency(model, sample_input)
    print(f"Latency (no skip): {latency['mean_ms']:.1f}ms, {latency['tokens_per_sec']:.0f} tok/s")

    # Skip threshold sweep
    print("\nSkip threshold sweep:")
    sweep_results = run_skip_sweep(model, val_loader, device, config)

    # Save all results
    final_results = {
        "config": {k: v for k, v in vars(config).items()},
        "n_params": n_params,
        "flops": flops,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(best_val_loss),
        "latency": latency,
        "skip_sweep": sweep_results,
    }
    save_results(final_results, args.output_dir, "final_results.json")

    return model


def train_looping(args):
    """Train looping AttnRes model."""
    train_loader, val_loader, actual_vocab = create_lm_dataloaders(
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_train=args.num_train,
        num_val=args.num_val,
    )

    config = LoopingTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_unique_blocks=args.n_unique_blocks,
        max_loop_depth=args.n_unique_blocks * args.max_loops,
        d_ff=args.d_model * 4,
        vocab_size=actual_vocab,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        attn_res_temperature=args.temperature,
        use_adaptive_halting=True,
        halt_threshold=args.halt_threshold,
        halt_penalty_weight=args.ponder_weight,
    )

    model = LoopingTransformerWithAttnRes(config)
    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in model.unique_blocks.parameters())
    device = torch.device(args.device)
    model = model.to(device)

    equiv_layers = args.n_unique_blocks * args.max_loops
    equiv_flops = count_transformer_flops(
        config.d_model, config.n_heads, config.d_ff,
        args.seq_len, equiv_layers, actual_vocab, args.batch_size,
    )

    print(f"{'='*60}")
    print(f"Training Looping AttnRes Model (ReLoop)")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Unique block parameters: {unique_params:,} ({unique_params/total_params:.1%})")
    print(f"Unique blocks: {args.n_unique_blocks}, Max loops: {args.max_loops}")
    print(f"Max equivalent depth: {equiv_layers} layers")
    print(f"Max FLOPs/batch: {equiv_flops['total_gflops']:.2f} GFLOPs")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = TrainLogger(args.output_dir, "train")

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_depth = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model.forward_with_loss(input_ids, labels)
                loss = result["loss"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(global_step)

            epoch_loss += loss.item()
            depth = result.get("effective_depth", 0)
            if isinstance(depth, torch.Tensor):
                depth = depth.item()
            epoch_depth += depth
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                ponder = result.get("ponder_cost", torch.tensor(0))
                if isinstance(ponder, torch.Tensor):
                    ponder = ponder.item()
                logger.log(
                    {"step": global_step, "loss": loss.item(),
                     "lm_loss": result["lm_loss"].item(),
                     "ponder_cost": ponder,
                     "effective_depth": depth,
                     "lr": optimizer.param_groups[0]["lr"]},
                    f"  step {global_step}: loss={loss.item():.4f} lm={result['lm_loss'].item():.4f} depth={depth:.1f}"
                )

        avg_loss = epoch_loss / n_batches
        avg_depth = epoch_depth / n_batches

        # Validate
        val_loss = evaluate_looping_model(model, val_loader, device)

        logger.log(
            {"epoch": epoch + 1, "train_loss": avg_loss,
             "val_loss": val_loss, "val_ppl": math.exp(min(val_loss, 20)),
             "avg_depth": avg_depth},
            f"Epoch {epoch+1}/{args.epochs}: train={avg_loss:.4f} val={val_loss:.4f} ppl={math.exp(min(val_loss, 20)):.2f} depth={avg_depth:.1f}"
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch, "global_step": global_step,
            "val_loss": val_loss,
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            print(f"  -> New best model (val_loss={val_loss:.4f})")

    logger.close()

    # Depth analysis
    print(f"\n{'='*60}")
    print("Depth Analysis by Input Difficulty")
    print(f"{'='*60}")
    analyze_looping_depth(model, actual_vocab, args.seq_len, device)

    final_results = {
        "config": {k: v for k, v in vars(config).items()},
        "total_params": total_params,
        "unique_params": unique_params,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
    }
    save_results(final_results, args.output_dir, "final_results.json")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(model, val_loader, device):
    """Evaluate standard model. Returns (loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        result = model(input_ids, labels=labels)
        # Weight by number of tokens
        n_tokens = (labels[:, 1:] != -100).sum().item()
        if n_tokens == 0:
            n_tokens = labels.shape[0] * (labels.shape[1] - 1)
        total_loss += result["loss"].item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    return avg_loss, ppl


@torch.no_grad()
def evaluate_looping_model(model, val_loader, device):
    """Evaluate looping model. Returns loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        result = model.forward_with_loss(input_ids, labels)
        total_loss += result["lm_loss"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def run_skip_sweep(model, val_loader, device, config):
    """Sweep skip thresholds and report accuracy vs FLOPs."""
    model.eval()
    thresholds = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results = []

    # Enable skipping
    old_skip = model.model.config.enable_skipping
    model.model.config.enable_skipping = True

    for eps in thresholds:
        model.model.config.skip_threshold = eps

        total_loss = 0.0
        total_blocks = 0
        total_tokens = 0
        n_batches = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            result = model(input_ids, labels=labels, return_routing_info=True)
            n_tokens = labels.shape[0] * (labels.shape[1] - 1)
            total_loss += result["loss"].item() * n_tokens
            total_tokens += n_tokens
            total_blocks += result.get("num_blocks_executed", config.n_blocks)
            n_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        avg_blocks = total_blocks / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        flops_ratio = avg_blocks / config.n_blocks

        # Effective FLOP count
        flop_info = count_effective_flops(config, val_loader.dataset[0]["input_ids"].shape[0], int(avg_blocks))

        entry = {
            "threshold": eps,
            "val_loss": avg_loss,
            "perplexity": ppl,
            "avg_blocks": avg_blocks,
            "total_blocks": config.n_blocks,
            "flops_ratio": flops_ratio,
            "effective_gflops": flop_info["effective_gflops"],
            "full_gflops": flop_info["full_gflops"],
        }
        results.append(entry)

        marker = " <-- best trade-off" if 0.01 <= eps <= 0.05 else ""
        print(f"  eps={eps:.3f}: loss={avg_loss:.4f} ppl={ppl:.2f} blocks={avg_blocks:.1f}/{config.n_blocks} FLOPs={flops_ratio:.1%}{marker}")

    model.model.config.enable_skipping = old_skip
    return results


@torch.no_grad()
def analyze_looping_depth(model, vocab_size, seq_len, device):
    """Analyze how looping depth varies by input difficulty."""
    model.eval()

    for diff_name, diff_val in [("easy", "easy"), ("medium", "mixed"), ("hard", "hard")]:
        ds = StructuredSyntheticLM(
            vocab_size=vocab_size, seq_len=seq_len,
            num_samples=200, difficulty_mix=diff_val, seed=999,
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        depths = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            result = model.forward_with_loss(input_ids, labels)
            if "effective_depth" in result:
                d = result["effective_depth"]
                if isinstance(d, torch.Tensor):
                    d = d.item()
                depths.append(d)

        avg_depth = sum(depths) / max(len(depths), 1)
        print(f"  {diff_name}: avg_depth={avg_depth:.2f}")


def evaluate_checkpoint(args):
    """Load checkpoint and run comprehensive evaluation."""
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt["config"]

    model = AdaptiveTransformerForCausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Use checkpoint's seq_len to avoid positional embedding OOB
    eval_seq_len = config.max_seq_len

    # Create validation data
    _, val_loader, _ = create_lm_dataloaders(
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        seq_len=eval_seq_len,
        vocab_size=config.vocab_size,
        num_train=100,  # minimal
        num_val=args.num_val,
    )

    print(f"{'='*60}")
    print(f"Evaluating checkpoint (epoch {ckpt.get('epoch', '?')})")
    print(f"{'='*60}")
    print(f"Config: d={config.d_model}, L={config.n_layers}, N={config.n_blocks}, seq={eval_seq_len}")

    # Base evaluation
    val_loss, val_ppl = evaluate_model(model, val_loader, device)
    print(f"Validation: loss={val_loss:.4f} ppl={val_ppl:.2f}")

    # Latency profiling
    sample = torch.randint(0, config.vocab_size, (1, eval_seq_len), device=device)
    latency_full = profile_latency(model, sample)
    print(f"\nLatency (full depth): {latency_full['mean_ms']:.1f}ms, {latency_full['tokens_per_sec']:.0f} tok/s")

    # Skip sweep
    print("\nSkip threshold sweep:")
    sweep = run_skip_sweep(model, val_loader, device, config)

    # Latency with best skip threshold
    best_entry = min(
        [e for e in sweep if e["perplexity"] < val_ppl * 1.02],
        key=lambda e: e["flops_ratio"],
        default=sweep[0],
    )

    if best_entry["threshold"] > 0:
        model.model.config.enable_skipping = True
        model.model.config.skip_threshold = best_entry["threshold"]
        latency_skip = profile_latency(model, sample)
        speedup = latency_full["mean_ms"] / latency_skip["mean_ms"]
        print(f"\nBest skip (eps={best_entry['threshold']}): "
              f"{latency_skip['mean_ms']:.1f}ms ({speedup:.2f}x speedup), "
              f"ppl={best_entry['perplexity']:.2f}")

    # Routing analysis
    print("\nRouting weight analysis:")
    model.model.config.enable_skipping = False
    stats = model.model.get_routing_statistics(sample)
    print(f"  Block importance scores: {[f'{x:.3f}' for x in stats['block_importance']]}")

    # Save
    results = {
        "val_loss": val_loss, "val_ppl": val_ppl,
        "latency_full": latency_full,
        "skip_sweep": sweep,
        "block_importance": stats["block_importance"],
    }
    save_results(results, args.output_dir, "eval_results.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate AttnRes adaptive transformer")
    parser.add_argument("--mode", choices=["standard", "looping", "eval"], default="standard")
    parser.add_argument("--config", type=str, default="", help="YAML config file")

    # Data
    parser.add_argument("--dataset", type=str, default="structured_synthetic",
                       choices=["structured_synthetic", "wikitext", "wikitext2"])
    parser.add_argument("--num_train", type=int, default=50000)
    parser.add_argument("--num_val", type=int, default=5000)

    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--seq_len", type=int, default=256)

    # Looping
    parser.add_argument("--n_unique_blocks", type=int, default=4)
    parser.add_argument("--max_loops", type=int, default=3)
    parser.add_argument("--halt_threshold", type=float, default=0.95)
    parser.add_argument("--ponder_weight", type=float, default=0.01)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Skipping
    parser.add_argument("--skip_threshold", type=float, default=0.01)

    # Eval
    parser.add_argument("--checkpoint", type=str, default="")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs/standard")
    parser.add_argument("--log_every", type=int, default=25)

    args = parser.parse_args()

    # Load YAML config if provided
    if args.config and os.path.exists(args.config):
        cfg = load_yaml_config(args.config)
        for key, value in cfg.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    if args.mode == "standard":
        train_standard(args)
    elif args.mode == "looping":
        args.output_dir = args.output_dir.replace("standard", "looping")
        train_looping(args)
    elif args.mode == "eval":
        if not args.checkpoint:
            # Try default location
            args.checkpoint = os.path.join(args.output_dir, "best.pt")
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
