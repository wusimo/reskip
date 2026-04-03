"""
Utilities: FLOP counting, latency profiling, config loading, logging.
"""

import json
import math
import os
import time
import yaml
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# FLOP Counting
# ---------------------------------------------------------------------------


def count_transformer_flops(
    d_model: int,
    n_heads: int,
    d_ff: int,
    seq_len: int,
    n_layers: int,
    vocab_size: int,
    batch_size: int = 1,
) -> dict:
    """
    Estimate FLOPs for a transformer forward pass.

    Returns per-component and total FLOPs.
    Follows Kaplan et al. (2020) scaling law conventions.
    """
    head_dim = d_model // n_heads

    # Per-layer FLOPs
    # Self-attention: QKV projection + attention scores + attention output + output projection
    qkv_flops = 3 * 2 * seq_len * d_model * d_model  # 3 projections
    attn_score_flops = 2 * n_heads * seq_len * seq_len * head_dim  # Q @ K^T
    attn_output_flops = 2 * n_heads * seq_len * seq_len * head_dim  # attn @ V
    out_proj_flops = 2 * seq_len * d_model * d_model

    attn_total = qkv_flops + attn_score_flops + attn_output_flops + out_proj_flops

    # SwiGLU FFN: gate + up + down (3 linear layers)
    ffn_flops = 3 * 2 * seq_len * d_model * d_ff

    layer_flops = attn_total + ffn_flops
    all_layers_flops = layer_flops * n_layers

    # Embedding + LM head
    emb_flops = 2 * seq_len * d_model * vocab_size  # embedding lookup is negligible, head is not

    total = (all_layers_flops + emb_flops) * batch_size

    return {
        "per_layer_attn": attn_total,
        "per_layer_ffn": ffn_flops,
        "per_layer_total": layer_flops,
        "all_layers": all_layers_flops,
        "embedding_and_head": emb_flops,
        "total": total,
        "total_gflops": total / 1e9,
        "n_layers": n_layers,
    }


def count_effective_flops(
    config,
    seq_len: int,
    blocks_executed: int,
    batch_size: int = 1,
) -> dict:
    """Count FLOPs with skipped blocks."""
    total_blocks = config.n_blocks
    layers_per_block = config.n_layers // total_blocks

    full = count_transformer_flops(
        config.d_model, config.n_heads, config.d_ff,
        seq_len, config.n_layers, config.vocab_size, batch_size,
    )

    effective_layers = blocks_executed * layers_per_block
    effective = count_transformer_flops(
        config.d_model, config.n_heads, config.d_ff,
        seq_len, effective_layers, config.vocab_size, batch_size,
    )

    return {
        "full_gflops": full["total_gflops"],
        "effective_gflops": effective["total_gflops"],
        "flops_ratio": effective["total"] / full["total"] if full["total"] > 0 else 1.0,
        "blocks_executed": blocks_executed,
        "total_blocks": total_blocks,
        "layers_executed": effective_layers,
        "total_layers": config.n_layers,
    }


# ---------------------------------------------------------------------------
# Latency Profiling
# ---------------------------------------------------------------------------


@torch.no_grad()
def profile_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 20,
    **forward_kwargs,
) -> dict:
    """
    Profile inference latency with warmup.

    Returns timing statistics in milliseconds.
    """
    model.eval()
    device = input_ids.device

    # Warmup
    for _ in range(num_warmup):
        _ = model(input_ids, **forward_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model(input_ids, **forward_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = sorted(times)
    return {
        "mean_ms": sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "min_ms": times[0],
        "max_ms": times[-1],
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "num_runs": num_runs,
        "batch_size": input_ids.shape[0],
        "seq_len": input_ids.shape[1],
        "tokens_per_sec": input_ids.shape[0] * input_ids.shape[1] / (sum(times) / len(times) / 1000),
    }


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config and flatten nested dicts."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Flatten one level of nesting
    flat = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            for k, v in value.items():
                flat[k] = v
        else:
            flat[key] = value

    return flat


def merge_args_with_config(args, config_path: Optional[str] = None) -> dict:
    """Merge CLI args with YAML config. CLI args take precedence."""
    if config_path and os.path.exists(config_path):
        cfg = load_yaml_config(config_path)
        # CLI args override config
        args_dict = vars(args)
        for key, value in cfg.items():
            if key not in args_dict or args_dict[key] is None:
                args_dict[key] = value
        return args_dict
    return vars(args)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TrainLogger:
    """Structured JSON logger with console output."""

    def __init__(self, log_dir: str, experiment_name: str = "train"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{experiment_name}_log.jsonl")
        self.log_file = open(self.log_path, "w")
        self.start_time = time.time()

    def log(self, entry: dict, print_msg: Optional[str] = None):
        """Log a dict entry and optionally print a message."""
        entry["timestamp"] = time.time() - self.start_time

        # Convert tensors to scalars
        for k, v in entry.items():
            if isinstance(v, torch.Tensor):
                entry[k] = v.item()

        self.log_file.write(json.dumps(entry) + "\n")
        self.log_file.flush()

        if print_msg:
            print(print_msg)

    def close(self):
        self.log_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def save_results(results: dict, output_dir: str, filename: str = "results.json"):
    """Save results dict to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Learning Rate Scheduler
# ---------------------------------------------------------------------------


class CosineWarmupScheduler:
    """Cosine LR with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, step):
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)
