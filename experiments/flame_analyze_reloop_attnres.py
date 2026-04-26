"""Analyze AttnRes-native reloop halting (v3).

Runs inference with multiple attnres_halt_threshold values and reports:
  1. Per-threshold depth histogram + LM loss (Pareto curve data)
  2. Per-position, per-sample routing signal statistics (mean_recent_weight)
  3. Cross-sample variance to assess dynamic-depth potential

Does NOT modify the model or any existing scripts.

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/flame_analyze_reloop_attnres.py \
    --model_path flame/saves/loopskip_transformer_test-v3 \
    --dataset /home/user01/Minko/datasets/fineweb_edu_100BT \
    --dataset_split train \
    --seq_len 65536 --context_len 2048 --batch_size 1 \
    --num_batches 32 --streaming --varlen \
    --device cuda:0 --dtype bf16 \
    --output_path outputs/reloop_v3_attnres_analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from flame_reskip_common import (
    build_text_dataloader,
    count_valid_tokens,
    load_model_and_tokenizer,
    save_json,
)
from fla.models.reloop_transformer.modeling_reloop_transformer import (
    collect_completed_blocks,
    per_sample_recent_weight,
)
from fla.ops.utils import prepare_position_ids


def build_position_ids(input_ids: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    if cu_seqlens is not None:
        return prepare_position_ids(cu_seqlens).to(torch.int32)
    return (
        torch.arange(0, input_ids.shape[1], device=input_ids.device)
        .repeat(input_ids.shape[0], 1)
        .to(torch.int32)
    )


# ── Phase 1: Collect per-sample routing signals ──


@torch.no_grad()
def collect_routing_signals(
    *,
    model,
    dataloader,
    device: str,
    num_batches: int,
) -> dict[str, Any]:
    """Run full-depth inference, collect per-sample per-position recent_weight."""
    config = model.config
    num_positions = int(config.attn_res_num_blocks)
    base_model = model.model  # ReLoopTransformerModel

    # Per-position lists of per-sample recent_weight values
    position_recent_weights: list[list[float]] = [[] for _ in range(num_positions)]
    losses: list[float] = []
    total_tokens = 0

    # Force full depth for signal collection (override halt to never trigger)
    saved_threshold = config.attnres_halt_threshold
    config.attnres_halt_threshold = -1.0  # impossible to halt

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        cu_seqlens = batch.get("cu_seqlens")
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device)
        position_ids = build_position_ids(input_ids, cu_seqlens)

        # Forward with full depth to get loss
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            return_routing_info=True,
        )
        valid_tokens = count_valid_tokens(labels)
        losses.append(float(outputs.loss.item()))
        total_tokens += valid_tokens

        # Now compute per-sample recent_weight at each position
        # We need to re-run the block forward to get block_states.
        # Instead, use the execution_trace which should have all 8 positions executed.
        # But per-sample recent_weight requires calling per_sample_recent_weight()
        # with the actual block_states. Let's do a lightweight forward to get them.
        inputs_embeds = base_model.embeddings(input_ids)
        block_states: list[torch.Tensor | None] = [None] * (num_positions + 1)
        block_states[0] = inputs_embeds
        hidden_states = inputs_embeds

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            if torch.all(attention_mask):
                attention_mask = None

        for position, block_idx in enumerate(base_model.block_schedule):
            current_block = base_model.layers[block_idx]
            # Compute per-sample recent weight for this position
            completed_blocks, completed_source_ids = collect_completed_blocks(
                block_states, position
            )
            routers_for_halt = []
            for layer in current_block.layers:
                routers_for_halt.extend([layer.attn_router, layer.mlp_router])

            if position > 0 and len(completed_blocks) > 1:
                recent_w = per_sample_recent_weight(routers_for_halt, completed_blocks)
                for sample_idx in range(recent_w.shape[0]):
                    position_recent_weights[position].append(float(recent_w[sample_idx].item()))
            else:
                for _ in range(inputs_embeds.shape[0]):
                    position_recent_weights[position].append(1.0)  # first position, no signal

            # Execute block to update block_states for next position
            hidden_states, _, _, _, _, _, _ = current_block(
                block_states=block_states,
                current_block_idx=position,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                cache_layer_offset=position * base_model.layers_per_block,
            )
            block_states[position + 1] = hidden_states

        if (batch_idx + 1) % 8 == 0:
            print(f"  collected {batch_idx + 1}/{num_batches} batches")

    config.attnres_halt_threshold = saved_threshold

    # Compute statistics
    position_stats = []
    for pos in range(num_positions):
        vals = position_recent_weights[pos]
        if vals:
            t = torch.tensor(vals)
            position_stats.append({
                "position": pos,
                "mean": float(t.mean()),
                "std": float(t.std()),
                "min": float(t.min()),
                "max": float(t.max()),
                "p25": float(t.quantile(0.25)),
                "p75": float(t.quantile(0.75)),
                "count": len(vals),
            })
        else:
            position_stats.append({"position": pos, "count": 0})

    avg_loss = sum(losses) / max(len(losses), 1)
    return {
        "full_depth_loss": avg_loss,
        "full_depth_ppl": math.exp(min(avg_loss, 20.0)),
        "total_tokens": total_tokens,
        "num_batches": len(losses),
        "position_recent_weight_stats": position_stats,
        "raw_recent_weights": {
            str(pos): vals for pos, vals in enumerate(position_recent_weights)
        },
    }


# ── Phase 2: Threshold sweep ──


@torch.no_grad()
def sweep_thresholds(
    *,
    model,
    dataloader,
    device: str,
    num_batches: int,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """For each threshold, run inference with actual halting and report depth + loss."""
    config = model.config
    num_positions = int(config.attn_res_num_blocks)
    results = []

    for thr in thresholds:
        config.attnres_halt_threshold = thr
        depths: list[float] = []
        losses: list[float] = []
        total_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            cu_seqlens = batch.get("cu_seqlens")
            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.to(device)
            position_ids = build_position_ids(input_ids, cu_seqlens)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                cu_seqlens=cu_seqlens,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                return_routing_info=True,
            )
            valid_tokens = count_valid_tokens(labels)
            losses.append(float(outputs.loss.item()) * valid_tokens)
            total_tokens += valid_tokens
            depths.append(float(outputs.routing_info["effective_depth"]))

        avg_loss = sum(losses) / max(total_tokens, 1)
        depth_hist = Counter(depths)
        entry = {
            "threshold": thr,
            "avg_loss": avg_loss,
            "ppl": math.exp(min(avg_loss, 20.0)),
            "avg_depth": sum(depths) / max(len(depths), 1),
            "unique_depths": len(set(depths)),
            "min_depth": min(depths) if depths else 0,
            "max_depth": max(depths) if depths else 0,
            "depth_std": float(torch.tensor(depths).std()) if len(depths) > 1 else 0.0,
            "depth_histogram": {f"{k:.0f}": v for k, v in sorted(depth_hist.items())},
            "num_batches": len(depths),
        }
        results.append(entry)
        print(f"  thr={thr:.3f} depth={entry['avg_depth']:.2f}±{entry['depth_std']:.2f} "
              f"unique={entry['unique_depths']} ppl={entry['ppl']:.2f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AttnRes-native reloop halting (v3).")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--data_files", default=None)
    parser.add_argument("--seq_len", type=int, default=65536)
    parser.add_argument("--context_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30",
        help="Comma-separated halt thresholds for sweep",
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, dtype=args.dtype)

    # Verify this is an attnres-halt model
    halt_mode = getattr(model.config, "halt_mode", "head")
    print(f"halt_mode={halt_mode}, training_full_depth={getattr(model.config, 'training_full_depth', False)}")
    if halt_mode != "attnres":
        print("WARNING: model halt_mode is not 'attnres'. Results may not reflect AttnRes-native halting.")

    thresholds = [float(x) for x in args.thresholds.split(",")]

    # Phase 1: collect per-sample routing signals at full depth
    print("\n=== Phase 1: Collecting per-sample routing signals (full depth) ===")
    dataloader1 = build_text_dataloader(
        tokenizer=tokenizer, dataset=args.dataset, dataset_name=args.dataset_name,
        dataset_split=args.dataset_split, data_dir=args.data_dir, data_files=args.data_files,
        seq_len=args.seq_len, context_len=args.context_len, batch_size=args.batch_size,
        num_workers=args.num_workers, streaming=args.streaming, varlen=args.varlen,
        seed=args.seed,
    )
    signals = collect_routing_signals(
        model=model, dataloader=dataloader1, device=args.device, num_batches=args.num_batches,
    )

    print(f"\nFull-depth loss={signals['full_depth_loss']:.4f} ppl={signals['full_depth_ppl']:.2f}")
    print("\nPer-position recent_weight stats:")
    print(f"{'pos':>4s} {'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s} {'p25':>7s} {'p75':>7s}")
    for s in signals["position_recent_weight_stats"]:
        if s["count"] > 0:
            print(f"  {s['position']:2d}  {s['mean']:.4f}  {s['std']:.4f}  "
                  f"{s['min']:.4f}  {s['max']:.4f}  {s['p25']:.4f}  {s['p75']:.4f}")

    # Phase 2: threshold sweep
    print("\n=== Phase 2: Threshold sweep ===")
    dataloader2 = build_text_dataloader(
        tokenizer=tokenizer, dataset=args.dataset, dataset_name=args.dataset_name,
        dataset_split=args.dataset_split, data_dir=args.data_dir, data_files=args.data_files,
        seq_len=args.seq_len, context_len=args.context_len, batch_size=args.batch_size,
        num_workers=args.num_workers, streaming=args.streaming, varlen=args.varlen,
        seed=args.seed,
    )
    sweep = sweep_thresholds(
        model=model, dataloader=dataloader2, device=args.device,
        num_batches=args.num_batches, thresholds=thresholds,
    )

    # Save
    result = {
        "signals": {k: v for k, v in signals.items() if k != "raw_recent_weights"},
        "threshold_sweep": sweep,
    }
    # Save raw weights separately (large)
    raw_path = Path(args.output_path).with_suffix(".raw_weights.json")
    save_json(str(raw_path), {"raw_recent_weights": signals["raw_recent_weights"]})

    save_json(args.output_path, result)
    print(f"\nSaved analysis to {args.output_path}")
    print(f"Saved raw weights to {raw_path}")


if __name__ == "__main__":
    main()
