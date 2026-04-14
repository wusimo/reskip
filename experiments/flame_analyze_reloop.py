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
from fla.ops.utils import prepare_position_ids


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def summarize_histogram(values: list[float], *, bin_size: float = 1.0) -> dict[str, int]:
    hist = Counter()
    for value in values:
        bucket = round(round(value / bin_size) * bin_size, 4)
        key = f"{bucket:.2f}"
        hist[key] += 1
    return dict(sorted(hist.items(), key=lambda item: float(item[0])))


def infer_hard_exit_depth(halt_probabilities: list[float], halt_threshold: float, num_positions: int) -> int:
    cumulative = 0.0
    for idx, probability in enumerate(halt_probabilities):
        cumulative += float(probability)
        if cumulative >= halt_threshold:
            return idx + 1
    return num_positions


def infer_hard_exit_depth_from_trace(execution_trace: list[dict[str, Any]], num_positions: int) -> int:
    for entry in execution_trace:
        if str(entry.get("status")) == "halted":
            return int(entry["position"])
    return num_positions


def build_position_ids(input_ids: torch.Tensor, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    if cu_seqlens is not None:
        return prepare_position_ids(cu_seqlens).to(torch.int32)
    return (
        torch.arange(0, input_ids.shape[1], device=input_ids.device)
        .repeat(input_ids.shape[0], 1)
        .to(torch.int32)
    )


@torch.no_grad()
def analyze_reloop(
    *,
    model,
    dataloader,
    device: str,
    num_batches: int,
    sample_trace_limit: int,
) -> dict[str, Any]:
    config = model.config
    num_positions = int(config.attn_res_num_blocks)
    halt_threshold = float(config.halt_threshold)

    total_loss = 0.0
    total_tokens = 0
    importance_matrix_sum = torch.zeros(num_positions, num_positions, dtype=torch.float32)
    block_importance_sum = torch.zeros(num_positions, dtype=torch.float32)
    self_importance_sum = torch.zeros(num_positions, dtype=torch.float32)
    halt_probability_sum = torch.zeros(num_positions, dtype=torch.float32)
    halt_probability_count = torch.zeros(num_positions, dtype=torch.float32)
    position_executed_fraction_sum = torch.zeros(num_positions, dtype=torch.float32)
    position_halt_probability_sum = torch.zeros(num_positions, dtype=torch.float32)
    position_routing_entropy_sum = torch.zeros(num_positions, dtype=torch.float32)
    position_recent_weight_sum = torch.zeros(num_positions, dtype=torch.float32)
    position_embed_weight_sum = torch.zeros(num_positions, dtype=torch.float32)
    position_seen_count = torch.zeros(num_positions, dtype=torch.float32)
    position_status_counts: list[Counter[str]] = [Counter() for _ in range(num_positions)]

    effective_depths: list[float] = []
    expected_depths: list[float] = []
    compute_ratios: list[float] = []
    ponder_costs: list[float] = []
    hard_exit_depths_from_trace: list[float] = []
    hard_exit_depths_from_probs: list[float] = []
    routing_entropies: list[float] = []
    sample_traces: list[dict[str, Any]] = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        cu_seqlens = batch.get("cu_seqlens")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            if torch.all(attention_mask):
                attention_mask = None
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device)
        position_ids = build_position_ids(input_ids, cu_seqlens)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            return_routing_info=True,
        )

        routing_info = outputs.routing_info
        if routing_info is None:
            raise RuntimeError("Expected routing_info for reloop analysis.")

        valid_tokens = count_valid_tokens(labels)
        total_loss += float(outputs.loss.item()) * valid_tokens
        total_tokens += valid_tokens

        effective_depth = float(routing_info["effective_depth"])
        expected_depth = float(routing_info.get("expected_depth", effective_depth))
        compute_ratio = float(routing_info.get("compute_ratio", 1.0))
        ponder_cost = float(routing_info.get("ponder_cost", effective_depth))
        halt_probabilities = list(routing_info.get("halt_probabilities") or [])
        execution_trace = list(routing_info.get("execution_trace") or [])

        effective_depths.append(effective_depth)
        expected_depths.append(expected_depth)
        compute_ratios.append(compute_ratio)
        ponder_costs.append(ponder_cost)
        if "routing_entropy" in routing_info:
            routing_entropies.append(float(routing_info["routing_entropy"]))
        hard_exit_depths_from_trace.append(float(infer_hard_exit_depth_from_trace(execution_trace, num_positions)))

        importance_matrix_sum += routing_info["importance_matrix"].float().cpu()
        block_importance_sum += torch.tensor(routing_info["block_importance"], dtype=torch.float32)
        self_importance_sum += torch.tensor(routing_info.get("self_importance", [0.0] * num_positions), dtype=torch.float32)

        for pos, probability in enumerate(halt_probabilities):
            if pos >= num_positions:
                break
            halt_probability_sum[pos] += float(probability)
            halt_probability_count[pos] += 1.0
        if halt_probabilities:
            hard_exit_depths_from_probs.append(
                float(infer_hard_exit_depth(halt_probabilities, halt_threshold, num_positions))
            )

        for entry in execution_trace:
            position = int(entry["position"])
            if not (0 <= position < num_positions):
                continue
            position_status_counts[position][str(entry.get("status", "unknown"))] += 1
            position_executed_fraction_sum[position] += float(entry.get("executed_fraction", 0.0))
            position_halt_probability_sum[position] += float(entry.get("halt_probability", 0.0))
            if entry.get("avg_phase1_entropy") is not None:
                position_routing_entropy_sum[position] += float(entry["avg_phase1_entropy"])
            if entry.get("avg_phase1_recent_weight") is not None:
                position_recent_weight_sum[position] += float(entry["avg_phase1_recent_weight"])
            if entry.get("avg_phase1_embed_weight") is not None:
                position_embed_weight_sum[position] += float(entry["avg_phase1_embed_weight"])
            position_seen_count[position] += 1.0

        if len(sample_traces) < sample_trace_limit:
            sample_traces.append(
                {
                    "batch_idx": batch_idx,
                    "loss": float(outputs.loss.item()),
                    "tokens": int(valid_tokens),
                    "effective_depth": effective_depth,
                    "expected_depth": expected_depth,
                    "compute_ratio": compute_ratio,
                    "ponder_cost": ponder_cost,
                    "halt_probabilities": halt_probabilities,
                    "execution_trace": execution_trace,
                }
            )

    analyzed_batches = len(effective_depths)
    if analyzed_batches == 0:
        raise RuntimeError("No batches were analyzed.")

    position_summary = []
    for position in range(num_positions):
        seen = max(float(position_seen_count[position].item()), 1.0)
        halt_seen = max(float(halt_probability_count[position].item()), 1.0)
        position_summary.append(
            {
                "position": position,
                "mean_halt_probability": float(halt_probability_sum[position].item() / halt_seen),
                "mean_executed_fraction": float(position_executed_fraction_sum[position].item() / seen),
                "mean_trace_halt_probability": float(position_halt_probability_sum[position].item() / seen),
                "mean_phase1_entropy": float(position_routing_entropy_sum[position].item() / seen),
                "mean_phase1_recent_weight": float(position_recent_weight_sum[position].item() / seen),
                "mean_phase1_embed_weight": float(position_embed_weight_sum[position].item() / seen),
                "status_counts": dict(position_status_counts[position]),
            }
        )

    avg_loss = total_loss / max(total_tokens, 1)
    result = {
        "model_path": getattr(model, "name_or_path", ""),
        "num_batches": analyzed_batches,
        "tokens": total_tokens,
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 20.0)),
        "config": {
            "attn_res_num_blocks": num_positions,
            "num_recurrent_blocks": int(getattr(config, "num_recurrent_blocks", 0)),
            "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0)),
            "halt_threshold": halt_threshold,
            "ponder_loss_weight": float(getattr(config, "ponder_loss_weight", 0.0)),
            "ponder_budget_start_step": int(getattr(config, "ponder_budget_start_step", 0)),
            "ponder_target_depth_ratio": float(getattr(config, "ponder_target_depth_ratio", 0.0)),
            "block_schedule": list(getattr(model.model, "block_schedule", [])),
        },
        "aggregate": {
            "effective_depth": summarize(effective_depths),
            "expected_depth": summarize(expected_depths),
            "compute_ratio": summarize(compute_ratios),
            "ponder_cost": summarize(ponder_costs),
            "routing_entropy": summarize(routing_entropies),
        },
        "histograms": {
            "effective_depth": summarize_histogram(effective_depths, bin_size=1.0),
            "expected_depth_0p25": summarize_histogram(expected_depths, bin_size=0.25),
            "hard_exit_depth_from_trace": summarize_histogram(hard_exit_depths_from_trace, bin_size=1.0),
            "hard_exit_depth_from_halt_probs": summarize_histogram(hard_exit_depths_from_probs, bin_size=1.0),
        },
        "mean_importance_matrix": (importance_matrix_sum / analyzed_batches).tolist(),
        "mean_block_importance": (block_importance_sum / analyzed_batches).tolist(),
        "mean_self_importance": (self_importance_sum / analyzed_batches).tolist(),
        "position_summary": position_summary,
        "sample_traces": sample_traces,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reloop halting and routing statistics on flame data.")
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
    parser.add_argument("--num_batches", type=int, default=128)
    parser.add_argument("--sample_trace_limit", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, dtype=args.dtype)
    dataloader = build_text_dataloader(
        tokenizer=tokenizer,
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        data_dir=args.data_dir,
        data_files=args.data_files,
        seq_len=args.seq_len,
        context_len=args.context_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        streaming=args.streaming,
        varlen=args.varlen,
        seed=args.seed,
    )
    analysis = analyze_reloop(
        model=model,
        dataloader=dataloader,
        device=args.device,
        num_batches=args.num_batches,
        sample_trace_limit=args.sample_trace_limit,
    )
    save_json(args.output_path, analysis)
    print(json.dumps(analysis["aggregate"], indent=2))
    print(f"saved_analysis={Path(args.output_path)}")


if __name__ == "__main__":
    main()
