"""
Training script for AttnRes language models with adaptive computation.

Supports:
1. Standard AttnRes training
2. Standard transformer baseline without AttnRes
3. Looping AttnRes training
4. Checkpoint evaluation with skip sweeps
"""

import argparse
import contextlib
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adaptive_transformer import (
    AdaptiveTransformerConfig,
    AdaptiveTransformerForCausalLM,
    StandardTransformerForCausalLM,
)
from src.looping_transformer import (
    LoopingTransformerConfig,
    LoopingTransformerWithAttnRes,
)
from src.data import create_lm_dataloaders, StructuredSyntheticLM
from src.utils import (
    CosineWarmupScheduler,
    TrainLogger,
    count_effective_flops,
    count_transformer_flops,
    load_yaml_config,
    profile_latency,
    save_results,
)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def maybe_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def init_distributed(args):
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1

    if not args.distributed:
        return

    backend = "nccl" if args.device.startswith("cuda") else "gloo"
    dist.init_process_group(backend=backend)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"


def cleanup_distributed():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def maybe_wrap_ddp(model, args):
    if not args.distributed:
        return model
    if str(args.device).startswith("cuda"):
        return DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    return DDP(model)


def sync_metrics(metrics: dict) -> dict:
    if not is_distributed():
        return metrics
    obj = [metrics]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def get_amp_dtype(amp_dtype: str):
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    return None


def create_standard_model(args, actual_vocab):
    config = AdaptiveTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_blocks=args.n_blocks,
        d_ff=args.d_model * 4,
        vocab_size=actual_vocab,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        use_rope=args.use_rope,
        rope_base=args.rope_base,
        attn_res_temperature=args.temperature,
        skip_threshold=args.skip_threshold,
        enable_skipping=False,
        use_attn_res=args.mode != "baseline",
    )

    if args.mode == "baseline":
        model = StandardTransformerForCausalLM(config)
    else:
        model = AdaptiveTransformerForCausalLM(config)
    return model, config


def create_data(args):
    return create_lm_dataloaders(
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_train=args.num_train,
        num_val=args.num_val,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        tokenizer_name=args.tokenizer_name,
        text_key=args.text_key,
        global_rank=args.rank,
        world_size=args.world_size,
        num_workers=args.num_workers,
    )


def train_standard(args):
    train_loader, val_loader, actual_vocab = create_data(args)
    model, config = create_standard_model(args, actual_vocab)
    n_params = sum(p.numel() for p in model.parameters())
    device = torch.device(args.device)
    model = model.to(device)
    model = maybe_wrap_ddp(model, args)

    flops = count_transformer_flops(
        config.d_model, config.n_heads, config.d_ff,
        args.seq_len, config.n_layers, actual_vocab, args.batch_size,
    )

    maybe_print(f"{'='*60}")
    maybe_print(f"Training {'Baseline Transformer' if args.mode == 'baseline' else 'Standard AttnRes Model'}")
    maybe_print(f"{'='*60}")
    maybe_print(f"Parameters: {n_params:,}")
    maybe_print(f"FLOPs/batch: {flops['total_gflops']:.2f} GFLOPs")
    maybe_print(f"Dataset: {args.dataset} (train={len(train_loader.dataset)}, val={len(val_loader.dataset)})")
    maybe_print(f"Vocab: {actual_vocab}, Seq len: {args.seq_len}")
    maybe_print(f"Blocks: {config.n_blocks}, Layers/block: {config.n_layers // config.n_blocks}")
    maybe_print(f"Device: {device}, world_size={args.world_size}, amp={args.amp_dtype}, grad_accum={args.grad_accum_steps}")
    maybe_print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.epochs
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_steps, total_steps)

    amp_dtype = get_amp_dtype(args.amp_dtype)
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp_dtype == "fp16" and device.type == "cuda"))

    logger = None
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        logger = TrainLogger(args.output_dir, "train")

    best_val_loss = float("inf")
    optimizer_steps = 0

    for epoch in range(args.epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_entropy = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            is_last_microbatch = batch_idx == len(train_loader) - 1
            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or is_last_microbatch

            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and not should_step
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    result = model(
                        input_ids,
                        labels=labels,
                        return_routing_info=(args.mode != "baseline"),
                    )
                    loss = result["loss"]
                    loss_to_backward = loss / args.grad_accum_steps

                scaler.scale(loss_to_backward).backward()

            epoch_loss += loss.item()
            entropy = result.get("routing_entropy", 0.0)
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.item()
            epoch_entropy += entropy
            n_batches += 1

            if not should_step:
                continue

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step(optimizer_steps)
            optimizer_steps += 1

            if logger is not None and optimizer_steps % args.log_every == 0:
                logger.log(
                    {
                        "step": optimizer_steps,
                        "train_loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "routing_entropy": entropy,
                        "epoch": epoch,
                    },
                    f"  step {optimizer_steps}: loss={loss.item():.4f} entropy={entropy:.4f} lr={optimizer.param_groups[0]['lr']:.2e}",
                )

            if args.max_steps > 0 and optimizer_steps >= args.max_steps:
                break

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_entropy = epoch_entropy / max(n_batches, 1)

        if is_main_process():
            val_loss, val_ppl = evaluate_model(unwrap_model(model), val_loader, device)
            metrics = {"val_loss": val_loss, "val_ppl": val_ppl}
        else:
            metrics = {"val_loss": 0.0, "val_ppl": 0.0}
        metrics = sync_metrics(metrics)
        val_loss = metrics["val_loss"]
        val_ppl = metrics["val_ppl"]

        if logger is not None:
            logger.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "routing_entropy": avg_entropy,
                },
                f"Epoch {epoch+1}/{args.epochs}: train={avg_train_loss:.4f} val={val_loss:.4f} ppl={val_ppl:.2f} entropy={avg_entropy:.4f}",
            )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if is_main_process():
            ckpt = {
                "model_state_dict": unwrap_model(model).state_dict(),
                "config": config,
                "epoch": epoch,
                "global_step": optimizer_steps,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }
            torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
                maybe_print(f"  -> New best model (val_loss={val_loss:.4f})")

        if args.max_steps > 0 and optimizer_steps >= args.max_steps:
            break

    if logger is not None:
        logger.close()

    base_model = unwrap_model(model)
    if not is_main_process():
        return base_model

    maybe_print(f"\n{'='*60}")
    maybe_print("Post-training Analysis")
    maybe_print(f"{'='*60}")

    sample_input = torch.randint(0, actual_vocab, (1, args.seq_len), device=device)
    latency = profile_latency(base_model, sample_input)
    maybe_print(f"Latency (no skip): {latency['mean_ms']:.1f}ms, {latency['tokens_per_sec']:.0f} tok/s")

    if args.mode != "baseline":
        maybe_print("\nSkip threshold sweep:")
        sweep_results = run_skip_sweep(base_model, val_loader, device, config)
    else:
        sweep_results = []

    final_results = {
        "config": {k: v for k, v in vars(config).items()},
        "n_params": n_params,
        "flops": flops,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "latency": latency,
        "skip_sweep": sweep_results,
    }
    save_results(final_results, args.output_dir, "final_results.json")
    return base_model


def train_looping(args):
    train_loader, val_loader, actual_vocab = create_data(args)

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
        use_rope=args.use_rope,
        rope_base=args.rope_base,
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
    model = maybe_wrap_ddp(model, args)

    equiv_layers = args.n_unique_blocks * args.max_loops
    equiv_flops = count_transformer_flops(
        config.d_model, config.n_heads, config.d_ff,
        args.seq_len, equiv_layers, actual_vocab, args.batch_size,
    )

    maybe_print(f"{'='*60}")
    maybe_print("Training Looping AttnRes Model (ReLoop)")
    maybe_print(f"{'='*60}")
    maybe_print(f"Total parameters: {total_params:,}")
    maybe_print(f"Unique block parameters: {unique_params:,} ({unique_params/total_params:.1%})")
    maybe_print(f"Unique blocks: {args.n_unique_blocks}, Max loops: {args.max_loops}")
    maybe_print(f"Max equivalent depth: {equiv_layers} layers")
    maybe_print(f"Max FLOPs/batch: {equiv_flops['total_gflops']:.2f} GFLOPs")
    maybe_print(f"Dataset: {args.dataset}")
    maybe_print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_steps, total_steps)

    amp_dtype = get_amp_dtype(args.amp_dtype)
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp_dtype == "fp16" and device.type == "cuda"))

    logger = None
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        logger = TrainLogger(args.output_dir, "train")

    best_val_loss = float("inf")
    optimizer_steps = 0

    for epoch in range(args.epochs):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_depth = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            is_last_microbatch = batch_idx == len(train_loader) - 1
            should_step = ((batch_idx + 1) % args.grad_accum_steps == 0) or is_last_microbatch

            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and not should_step
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    loop_model = unwrap_model(model)
                    result = loop_model.forward_with_loss(input_ids, labels)
                    loss = result["loss"]
                    loss_to_backward = loss / args.grad_accum_steps

                scaler.scale(loss_to_backward).backward()

            epoch_loss += loss.item()
            depth = result.get("effective_depth", 0)
            if isinstance(depth, torch.Tensor):
                depth = depth.item()
            epoch_depth += depth
            n_batches += 1

            if not should_step:
                continue

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step(optimizer_steps)
            optimizer_steps += 1

            if logger is not None and optimizer_steps % args.log_every == 0:
                ponder = result.get("ponder_cost", torch.tensor(0))
                if isinstance(ponder, torch.Tensor):
                    ponder = ponder.item()
                logger.log(
                    {
                        "step": optimizer_steps,
                        "loss": loss.item(),
                        "lm_loss": result["lm_loss"].item(),
                        "ponder_cost": ponder,
                        "effective_depth": depth,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    f"  step {optimizer_steps}: loss={loss.item():.4f} lm={result['lm_loss'].item():.4f} depth={depth:.1f}",
                )

            if args.max_steps > 0 and optimizer_steps >= args.max_steps:
                break

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_depth = epoch_depth / max(n_batches, 1)

        if is_main_process():
            val_loss = evaluate_looping_model(unwrap_model(model), val_loader, device)
            metrics = {"val_loss": val_loss}
        else:
            metrics = {"val_loss": 0.0}
        metrics = sync_metrics(metrics)
        val_loss = metrics["val_loss"]

        if logger is not None:
            logger.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "val_ppl": math.exp(min(val_loss, 20)),
                    "avg_depth": avg_depth,
                },
                f"Epoch {epoch+1}/{args.epochs}: train={avg_loss:.4f} val={val_loss:.4f} ppl={math.exp(min(val_loss, 20)):.2f} depth={avg_depth:.1f}",
            )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if is_main_process():
            ckpt = {
                "model_state_dict": unwrap_model(model).state_dict(),
                "config": config,
                "epoch": epoch,
                "global_step": optimizer_steps,
                "val_loss": val_loss,
            }
            torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
                maybe_print(f"  -> New best model (val_loss={val_loss:.4f})")

        if args.max_steps > 0 and optimizer_steps >= args.max_steps:
            break

    if logger is not None:
        logger.close()

    if is_main_process():
        maybe_print(f"\n{'='*60}")
        maybe_print("Depth Analysis by Input Difficulty")
        maybe_print(f"{'='*60}")
        analyze_looping_depth(unwrap_model(model), actual_vocab, args.seq_len, device)
        final_results = {
            "config": {k: v for k, v in vars(config).items()},
            "total_params": total_params,
            "unique_params": unique_params,
            "best_val_loss": best_val_loss,
            "best_val_ppl": math.exp(min(best_val_loss, 20)),
        }
        save_results(final_results, args.output_dir, "final_results.json")

    return unwrap_model(model)


@torch.no_grad()
def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        result = model(input_ids, labels=labels)
        n_tokens = (labels[:, 1:] != -100).sum().item()
        if n_tokens == 0:
            n_tokens = labels.shape[0] * (labels.shape[1] - 1)
        total_loss += result["loss"].item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(min(avg_loss, 20))


@torch.no_grad()
def evaluate_looping_model(model, val_loader, device):
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
    model.eval()
    thresholds = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results = []

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
        flop_info = count_effective_flops(config, config.max_seq_len, int(avg_blocks))

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
        maybe_print(
            f"  eps={eps:.3f}: loss={avg_loss:.4f} ppl={ppl:.2f} "
            f"blocks={avg_blocks:.1f}/{config.n_blocks} FLOPs={flops_ratio:.1%}{marker}"
        )

    model.model.config.enable_skipping = old_skip
    return results


@torch.no_grad()
def analyze_looping_depth(model, vocab_size, seq_len, device):
    model.eval()

    for diff_name, diff_val in [("easy", "easy"), ("medium", "mixed"), ("hard", "hard")]:
        ds = StructuredSyntheticLM(
            vocab_size=vocab_size, seq_len=seq_len, num_samples=200, difficulty_mix=diff_val, seed=999,
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        depths = []

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            result = model.forward_with_loss(input_ids, labels)
            if "effective_depth" in result:
                depth = result["effective_depth"]
                if isinstance(depth, torch.Tensor):
                    depth = depth.item()
                depths.append(depth)

        maybe_print(f"  {diff_name}: avg_depth={sum(depths) / max(len(depths), 1):.2f}")


def evaluate_checkpoint(args):
    maybe_print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ckpt["config"]

    model = AdaptiveTransformerForCausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    _, val_loader, _ = create_lm_dataloaders(
        dataset_type=args.dataset,
        batch_size=args.batch_size,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        num_train=100,
        num_val=args.num_val,
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        tokenizer_name=args.tokenizer_name,
        text_key=args.text_key,
        num_workers=args.num_workers,
    )

    maybe_print(f"{'='*60}")
    maybe_print(f"Evaluating checkpoint (epoch {ckpt.get('epoch', '?')})")
    maybe_print(f"{'='*60}")
    maybe_print(f"Config: d={config.d_model}, L={config.n_layers}, N={config.n_blocks}, seq={config.max_seq_len}")

    val_loss, val_ppl = evaluate_model(model, val_loader, device)
    maybe_print(f"Validation: loss={val_loss:.4f} ppl={val_ppl:.2f}")

    sample = torch.randint(0, config.vocab_size, (1, config.max_seq_len), device=device)
    latency_full = profile_latency(model, sample)
    maybe_print(f"\nLatency (full depth): {latency_full['mean_ms']:.1f}ms, {latency_full['tokens_per_sec']:.0f} tok/s")

    maybe_print("\nSkip threshold sweep:")
    sweep = run_skip_sweep(model, val_loader, device, config)

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
        maybe_print(
            f"\nBest skip (eps={best_entry['threshold']}): "
            f"{latency_skip['mean_ms']:.1f}ms ({speedup:.2f}x speedup), "
            f"ppl={best_entry['perplexity']:.2f}"
        )

    maybe_print("\nRouting weight analysis:")
    model.model.config.enable_skipping = False
    stats = model.model.get_routing_statistics(sample)
    maybe_print(f"  Block importance scores: {[f'{x:.3f}' for x in stats['block_importance']]}")

    save_results(
        {
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "latency_full": latency_full,
            "skip_sweep": sweep,
            "block_importance": stats["block_importance"],
        },
        args.output_dir,
        "eval_results.json",
    )


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate AttnRes adaptive transformer")
    parser.add_argument("--mode", choices=["standard", "baseline", "looping", "eval"], default="standard")
    parser.add_argument("--config", type=str, default="", help="YAML config file")

    parser.add_argument(
        "--dataset",
        type=str,
        default="structured_synthetic",
        choices=["structured_synthetic", "wikitext", "wikitext2", "local_text", "slimpajama"],
    )
    parser.add_argument("--num_train", type=int, default=50000)
    parser.add_argument("--num_val", type=int, default=5000)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--val_data_path", type=str, default="")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--rope_base", type=float, default=10000.0)

    parser.add_argument("--n_unique_blocks", type=int, default=4)
    parser.add_argument("--max_loops", type=int, default=3)
    parser.add_argument("--halt_threshold", type=float, default=0.95)
    parser.add_argument("--ponder_weight", type=float, default=0.01)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--amp_dtype", choices=["none", "bf16", "fp16"], default="bf16")

    parser.add_argument("--skip_threshold", type=float, default=0.01)
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="outputs/standard")
    parser.add_argument("--log_every", type=int, default=25)

    args = parser.parse_args()
    init_distributed(args)

    if args.config and os.path.exists(args.config):
        cfg = load_yaml_config(args.config)
        for key, value in cfg.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    try:
        if args.mode in {"standard", "baseline"}:
            train_standard(args)
        elif args.mode == "looping":
            args.output_dir = args.output_dir.replace("standard", "looping")
            train_looping(args)
        elif args.mode == "eval":
            if not args.checkpoint:
                args.checkpoint = os.path.join(args.output_dir, "best.pt")
            evaluate_checkpoint(args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
