from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from flame_reskip_common import (
    count_valid_tokens,
    evaluate_causal_lm,
    load_model_and_tokenizer,
    parse_csv_floats,
    save_json,
    build_text_dataloader,
)


def dynamic_metric_from_trace(entry: dict, strategy: str) -> float | None:
    if strategy.startswith("prev_"):
        strategy = strategy[5:]
    recent = entry.get("avg_phase1_recent_weight")
    embed = entry.get("avg_phase1_embed_weight")
    entropy = entry.get("avg_phase1_entropy")
    if strategy == "recent_weight_gt":
        return recent
    if strategy == "recent_weight_lt":
        return recent
    if strategy == "embed_weight_gt":
        return embed
    if strategy == "entropy_lt":
        return entropy
    if strategy == "recent_minus_embed_gt":
        if recent is None or embed is None:
            return None
        return float(recent) - float(embed)
    if strategy == "recent_over_embed_gt":
        if recent is None or embed is None or float(embed) <= 1e-8:
            return None
        return float(recent) / float(embed)
    if strategy == "recent_confidence_gt":
        if recent is None or entropy is None:
            return None
        return float(recent) * max(1.0 - float(entropy), 0.0)
    if strategy == "recent_margin_confidence_gt":
        if recent is None or embed is None or entropy is None:
            return None
        return (float(recent) - float(embed)) * max(1.0 - float(entropy), 0.0)
    if strategy == "recent_x_entropy_lt":
        if recent is None or entropy is None:
            return None
        return float(recent) * max(1.0 - float(entropy), 0.0)
    raise ValueError(f"Unsupported dynamic skip strategy: {strategy}")


def dynamic_disable_threshold(strategy: str) -> float:
    if strategy.endswith("_lt"):
        return -1e9
    return 1e9


def dynamic_probe_cost_rank(probe_mode: str) -> int:
    order = {
        "cached_prev": 0,
        "first_attn": 0,
        "attn_only": 1,
        "first_layer": 2,
        "all": 3,
    }
    return order.get(probe_mode, 99)


def summarize_dynamic_metric_coverage(
    per_position_values: list[list[float]],
) -> dict[str, float | int]:
    covered_positions = sum(1 for values in per_position_values if values)
    num_values = sum(len(values) for values in per_position_values)
    return {
        "covered_positions": covered_positions,
        "num_values": num_values,
    }


def build_dynamic_position_modes(
    *,
    num_positions: int,
    block_importance: list[float],
    block_ablation_ppl: list[float] | None = None,
    mode_spec: str,
) -> list[tuple[str, list[int]]]:
    interior = [idx for idx in range(1, max(num_positions - 1, 1))]
    if not mode_spec.strip():
        return [("all", interior)]

    by_importance = sorted(interior, key=lambda idx: block_importance[idx])
    # Ablation-informed: sort interior blocks by static-removal PPL impact (lowest = safest to skip).
    # When ablation data is unavailable, fall back to reverse importance (highest importance = lowest
    # downstream weight = most likely to be compensated by other blocks).
    if block_ablation_ppl is not None:
        by_ablation = sorted(interior, key=lambda idx: block_ablation_ppl[idx] if idx < len(block_ablation_ppl) else float("inf"))
    else:
        by_ablation = sorted(interior, key=lambda idx: block_importance[idx], reverse=True)
    tail = interior[len(interior) // 2 :]
    tail_by_importance = sorted(tail, key=lambda idx: block_importance[idx])
    median_importance = None
    if interior:
        ordered = sorted(block_importance[idx] for idx in interior)
        median_importance = ordered[len(ordered) // 2]
    modes: list[tuple[str, list[int]]] = []
    for raw in mode_spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "auto":
            for expanded in ("recommended", "all", "low1", "low2", "low3", "late2", "late3", "taillow1", "taillow2", "taillow3", "ablation1", "ablation2", "ablation3"):
                raw = expanded
                token = raw.strip()
                if token in {name for name, _ in modes}:
                    continue
                if token == "recommended":
                    positions = [idx for idx in tail if median_importance is not None and block_importance[idx] <= median_importance]
                    if not positions and tail_by_importance:
                        positions = tail_by_importance[:1]
                    modes.append((token, positions))
                    continue
                if token == "all":
                    modes.append(("all", interior))
                    continue
                if token.startswith("low"):
                    count = int(token[3:])
                    positions = by_importance[: max(0, min(count, len(by_importance)))]
                    modes.append((token, positions))
                    continue
                if token.startswith("late"):
                    count = int(token[4:])
                    positions = interior[-max(0, min(count, len(interior))) :]
                    modes.append((token, positions))
                    continue
                if token.startswith("taillow"):
                    count = int(token[7:])
                    positions = sorted(tail_by_importance[: max(0, min(count, len(tail_by_importance)))])
                    modes.append((token, positions))
                    continue
                if token.startswith("ablation"):
                    count = int(token[8:])
                    positions = sorted(by_ablation[: max(0, min(count, len(by_ablation)))])
                    modes.append((token, positions))
                    continue
            continue
        if token == "all":
            modes.append(("all", interior))
            continue
        if token == "recommended":
            positions = [idx for idx in tail if median_importance is not None and block_importance[idx] <= median_importance]
            if not positions and tail_by_importance:
                positions = tail_by_importance[:1]
            modes.append((token, positions))
            continue
        if token.startswith("low"):
            count = int(token[3:])
            positions = by_importance[: max(0, min(count, len(by_importance)))]
            modes.append((token, positions))
            continue
        if token.startswith("late"):
            count = int(token[4:])
            positions = interior[-max(0, min(count, len(interior))) :]
            modes.append((token, positions))
            continue
        if token.startswith("taillow"):
            count = int(token[7:])
            positions = sorted(tail_by_importance[: max(0, min(count, len(tail_by_importance)))])
            modes.append((token, positions))
            continue
        if token.startswith("ablation"):
            count = int(token[8:])
            positions = sorted(by_ablation[: max(0, min(count, len(by_ablation)))])
            modes.append((token, positions))
            continue
        if token.startswith("custom:"):
            positions = [int(item) for item in token.split(":", 1)[1].split("|") if item.strip()]
            positions = [idx for idx in positions if 0 < idx < num_positions - 1]
            modes.append((token, positions))
            continue
        raise ValueError(f"Unsupported dynamic position mode: {token}")
    if not modes:
        modes.append(("all", interior))
    return modes


def expand_block_importance_to_layers(
    block_importance: list[float],
    layers_per_block: int,
) -> list[float]:
    if layers_per_block <= 0:
        return []
    expanded: list[float] = []
    for value in block_importance:
        expanded.extend([float(value)] * layers_per_block)
    return expanded


def select_fast_latency_candidates(
    dynamic_results: list[dict],
    *,
    full_perplexity: float,
    speed_ppl_tolerance: float,
    granularity: str,
    top_k: int,
) -> list[dict]:
    candidates = [
        item
        for item in dynamic_results
        if item["perplexity"] <= full_perplexity * (1.0 + speed_ppl_tolerance)
    ]
    if not candidates:
        return []

    depth_key_name = "avg_compute_units" if granularity == "mlp" else "avg_blocks"

    def item_key(item: dict) -> tuple[float, int, float]:
        return (
            float(item.get(depth_key_name, float("inf"))),
            dynamic_probe_cost_rank(item.get("probe_mode", "all")),
            float(item["perplexity"]),
        )

    frontier: list[dict] = []
    probe_modes = sorted({item.get("probe_mode", "all") for item in candidates}, key=dynamic_probe_cost_rank)
    for probe_mode in probe_modes:
        probe_items = sorted(
            [item for item in candidates if item.get("probe_mode", "all") == probe_mode],
            key=lambda item: (
                float(item.get(depth_key_name, float("inf"))),
                float(item["perplexity"]),
            ),
        )
        best_ppl_so_far = float("inf")
        for item in probe_items:
            ppl = float(item["perplexity"])
            if ppl + 1e-9 < best_ppl_so_far:
                frontier.append(item)
                best_ppl_so_far = ppl
        if probe_items:
            frontier.append(min(probe_items, key=lambda item: float(item["perplexity"])))
            frontier.append(min(probe_items, key=lambda item: float(item.get(depth_key_name, float("inf")))))

    unique_candidates: dict[tuple, dict] = {}
    for item in frontier:
        key = (
            item.get("probe_mode", "all"),
            item.get("position_mode", "unknown"),
            float(item.get("quantile", 0.0)),
            int(item.get("max_skips", 0)),
        )
        current = unique_candidates.get(key)
        if current is None or item_key(item) < item_key(current):
            unique_candidates[key] = item

    ranked = sorted(unique_candidates.values(), key=item_key)
    return ranked[: max(top_k, 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze routing and export skip-ready ReSkip flame checkpoints.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_split", default="validation")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--data_files", default=None)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--context_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=128)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--thresholds", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    parser.add_argument("--percentiles", default="50,60,70,80,90,95")
    parser.add_argument("--ppl_tolerance", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--output_dir", default="outputs/flame_analysis")
    parser.add_argument("--export_best_model_dir", default="")
    parser.add_argument("--export_best_dynamic_model_dir", default="")
    parser.add_argument("--dynamic_skip_strategy", default="")
    parser.add_argument(
        "--dynamic_skip_granularity",
        choices=("block",),
        default="block",
        help="Dynamic skip granularity (block-level only).",
    )
    parser.add_argument(
        "--dynamic_skip_probe_modes",
        default="all,first_attn",
        help="Comma-separated probe modes for dynamic skip: all, attn_only, first_layer, first_attn",
    )
    parser.add_argument("--dynamic_skip_quantiles", default="0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.99")
    parser.add_argument("--dynamic_skip_max_skips_options", default="1,2,3")
    parser.add_argument(
        "--dynamic_skip_position_modes",
        default="auto",
        help=(
            "Comma-separated dynamic skip candidate position sets. "
            "Supported tokens: auto, recommended, all, lowK, lateK, taillowK, custom:i|j|k"
        ),
    )
    parser.add_argument("--dynamic_skip_calibration_batches", type=int, default=0)
    parser.add_argument("--dynamic_skip_calibration_seed", type=int, default=0)
    parser.add_argument("--dynamic_skip_eval_seed", type=int, default=1)
    parser.add_argument("--dynamic_skip_speed_ppl_tolerance", type=float, default=0.05)
    parser.add_argument("--dynamic_skip_latency_num_batches", type=int, default=16)
    parser.add_argument("--dynamic_skip_latency_top_k", type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, dtype=args.dtype)
    latency_batches_cache: list[dict[str, torch.Tensor]] | None = None

    def run_eval(
        *,
        return_routing_info: bool,
        enable_skipping: bool | None,
        skip_keep_mask=None,
        return_batch_metrics: bool = False,
        seed: int = 0,
        num_batches: int | None = None,
        **model_forward_kwargs,
    ):
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
            seed=seed,
        )
        return evaluate_causal_lm(
            model,
            dataloader,
            args.device,
            num_batches=args.num_batches if num_batches is None else num_batches,
            return_routing_info=return_routing_info,
            enable_skipping=enable_skipping,
            skip_keep_mask=skip_keep_mask,
            return_batch_metrics=return_batch_metrics,
            **model_forward_kwargs,
        )

    def run_latency_eval(
        *,
        seed: int = 0,
        num_batches: int | None = None,
        **model_forward_kwargs,
    ):
        nonlocal latency_batches_cache
        if latency_batches_cache is None:
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
                seed=seed,
            )
            latency_batches_cache = []
            for batch_idx, batch in enumerate(dataloader):
                latency_batches_cache.append(batch)
                if num_batches is not None and (batch_idx + 1) >= num_batches:
                    break
        total_loss = 0.0
        total_tokens = 0
        total_time = 0.0
        measured_batches = 0
        warmup_batches = 2
        measured_total = num_batches if num_batches is not None else len(latency_batches_cache)
        for batch_idx, batch in enumerate(latency_batches_cache[:measured_total]):
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            attention_mask = batch.get("attention_mask")
            cu_seqlens = batch.get("cu_seqlens")
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)
                if torch.all(attention_mask):
                    attention_mask = None
            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.to(args.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cu_seqlens=cu_seqlens,
                use_cache=False,
                return_dict=True,
                return_routing_info=False,
                enable_skipping=False,
                skip_keep_mask=None,
                **model_forward_kwargs,
            )
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            valid_tokens = count_valid_tokens(labels)
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens
            if batch_idx >= warmup_batches:
                total_time += duration
                measured_batches += 1
        avg_loss = total_loss / max(total_tokens, 1)
        mean_batch_s = total_time / max(measured_batches, 1)
        return {
            "loss": avg_loss,
            "perplexity": math.exp(min(avg_loss, 20)),
            "tokens": total_tokens,
            "num_batches": measured_total,
            "measured_batches": measured_batches,
            "mean_batch_s": mean_batch_s,
            "tokens_per_s": (total_tokens * measured_batches / max(measured_total, 1)) / max(total_time, 1e-8),
        }

    decoder = model.get_decoder()
    decoder.clear_skip_keep_mask()
    full_eval = run_eval(
        return_routing_info=True,
        enable_skipping=False,
        return_batch_metrics=True,
    )

    thresholds = parse_csv_floats(args.thresholds)
    percentile_values = parse_csv_floats(args.percentiles)
    sweep = []
    for threshold in thresholds:
        keep_mask = decoder.build_keep_mask_from_importance(full_eval["block_importance"], threshold)
        metrics = run_eval(
            return_routing_info=True,
            enable_skipping=True,
            skip_keep_mask=keep_mask,
        )
        metrics["threshold"] = threshold
        metrics["keep_mask"] = keep_mask
        sweep.append(metrics)

    interior_importance = full_eval["block_importance"][:-1]
    if interior_importance and max(thresholds) <= min(interior_importance):
        print(
            "Warning: all analyzed thresholds are below the minimum interior block importance. "
            "The sweep cannot recommend skipping any middle blocks; try a higher threshold range."
        )

    percentile_sweep = []
    if interior_importance:
        interior_tensor = torch.tensor(interior_importance, dtype=torch.float32)
        for percentile in percentile_values:
            threshold = float(torch.quantile(interior_tensor, percentile / 100.0).item())
            keep_mask = decoder.build_keep_mask_from_importance(full_eval["block_importance"], threshold)
            metrics = run_eval(
                return_routing_info=True,
                enable_skipping=True,
                skip_keep_mask=keep_mask,
            )
            metrics["percentile"] = percentile
            metrics["threshold"] = threshold
            metrics["keep_mask"] = keep_mask
            percentile_sweep.append(metrics)

    block_ablation = []
    for block_idx in range(1, max(len(full_eval["block_importance"]) - 1, 1)):
        keep_mask = [True] * len(full_eval["block_importance"])
        keep_mask[block_idx] = False
        metrics = run_eval(
            return_routing_info=True,
            enable_skipping=True,
            skip_keep_mask=keep_mask,
        )
        metrics["ablated_block"] = block_idx
        metrics["keep_mask"] = keep_mask
        metrics["delta_ppl_pct"] = (metrics["perplexity"] / full_eval["perplexity"] - 1.0) * 100.0
        block_ablation.append(metrics)

    difficulty_buckets = []
    batch_metrics = full_eval.get("batch_metrics", [])
    if batch_metrics:
        sorted_batches = sorted(batch_metrics, key=lambda item: item["loss"])
        bucket_size = max(math.ceil(len(sorted_batches) / 3), 1)
        for bucket_name, bucket_idx in (("easy", 0), ("medium", 1), ("hard", 2)):
            start = bucket_idx * bucket_size
            end = min((bucket_idx + 1) * bucket_size, len(sorted_batches))
            bucket = sorted_batches[start:end]
            if not bucket:
                continue
            avg_loss = sum(item["loss"] for item in bucket) / len(bucket)
            avg_blocks = sum(item.get("avg_blocks", 0.0) for item in bucket) / len(bucket)
            importance_rows = [item["block_importance"] for item in bucket if "block_importance" in item]
            avg_importance = None
            if importance_rows:
                avg_importance = [
                    sum(row[i] for row in importance_rows) / len(importance_rows)
                    for i in range(len(importance_rows[0]))
                ]
            difficulty_buckets.append(
                {
                    "bucket": bucket_name,
                    "num_batches": len(bucket),
                    "tokens": sum(item["tokens"] for item in bucket),
                    "avg_loss": avg_loss,
                    "avg_perplexity": math.exp(min(avg_loss, 20)),
                    "avg_blocks": avg_blocks,
                    "avg_block_importance": avg_importance,
                }
            )

    dynamic_skip_analysis = None
    if args.dynamic_skip_strategy:
        dynamic_trace_field = "execution_trace"
        calibration_batches = (
            args.dynamic_skip_calibration_batches
            if args.dynamic_skip_calibration_batches > 0
            else args.num_batches
        )
        dynamic_num_positions = decoder.num_block_positions
        disabled_thresholds = [dynamic_disable_threshold(args.dynamic_skip_strategy)] * dynamic_num_positions
        dynamic_probe_modes = [item.strip() for item in args.dynamic_skip_probe_modes.split(",") if item.strip()]
        if args.dynamic_skip_strategy.startswith("prev_"):
            dynamic_probe_modes = ["cached_prev"]
        if not dynamic_probe_modes:
            raise ValueError("`--dynamic_skip_probe_modes` must contain at least one probe mode.")
        calibration_by_probe_mode = {}
        num_positions = 0
        for probe_mode in dynamic_probe_modes:
            probe_calibration = run_eval(
                return_routing_info=True,
                enable_skipping=False,
                return_batch_metrics=True,
                seed=args.dynamic_skip_calibration_seed,
                num_batches=calibration_batches,
                dynamic_skip_strategy=args.dynamic_skip_strategy,
                dynamic_skip_granularity=args.dynamic_skip_granularity,
                dynamic_skip_probe_mode=probe_mode,
                dynamic_skip_position_thresholds=disabled_thresholds,
                dynamic_skip_max_skips=0,
            )
            probe_num_positions = dynamic_num_positions
            num_positions = max(num_positions, probe_num_positions)
            per_position_values: list[list[float]] = [[] for _ in range(probe_num_positions)]
            for batch in probe_calibration.get("batch_metrics", []):
                for trace_entry in batch.get(dynamic_trace_field, []):
                    position = int(trace_entry["position"])
                    if position >= probe_num_positions:
                        continue
                    value = dynamic_metric_from_trace(trace_entry, args.dynamic_skip_strategy)
                    if value is None:
                        continue
                    per_position_values[position].append(float(value))
            calibration_by_probe_mode[probe_mode] = {
                "calibration_eval": probe_calibration,
                "per_position_values": per_position_values,
                "metric_coverage": summarize_dynamic_metric_coverage(per_position_values),
            }

        reference_probe_mode = dynamic_probe_modes[0]
        reference_calibration = calibration_by_probe_mode[reference_probe_mode]["calibration_eval"]
        dynamic_results = []
        dynamic_quantiles = parse_csv_floats(args.dynamic_skip_quantiles)
        dynamic_max_skips_options = [
            int(item.strip()) for item in args.dynamic_skip_max_skips_options.split(",") if item.strip()
        ]
        # Use block ablation PPL data (if available) for ablation-informed position modes.
        ablation_ppl_map = None
        if block_ablation:
            ablation_ppl_map = [0.0] * num_positions
            for entry in block_ablation:
                ablated = entry.get("ablated_block")
                if ablated is not None and ablated < num_positions:
                    ablation_ppl_map[ablated] = entry.get("perplexity", float("inf"))
        position_modes = build_dynamic_position_modes(
            num_positions=num_positions,
            block_importance=reference_calibration.get("block_importance", []),
            block_ablation_ppl=ablation_ppl_map,
            mode_spec=args.dynamic_skip_position_modes,
        )
        dynamic_eval_full = run_eval(
            return_routing_info=True,
            enable_skipping=False,
            return_batch_metrics=True,
            seed=args.dynamic_skip_eval_seed,
            dynamic_skip_strategy=args.dynamic_skip_strategy,
            dynamic_skip_granularity=args.dynamic_skip_granularity,
            dynamic_skip_position_thresholds=disabled_thresholds,
            dynamic_skip_max_skips=0,
        )

        def dynamic_depth_key(item: dict) -> float:
            if args.dynamic_skip_granularity == "mlp":
                return item.get("avg_compute_units", float("inf"))
            return item.get("avg_blocks", float("inf"))

        for probe_mode in dynamic_probe_modes:
            probe_calibration = calibration_by_probe_mode[probe_mode]
            per_position_values = probe_calibration["per_position_values"]
            for mode_name, allowed_positions in position_modes:
                allowed_set = set(allowed_positions)
                for max_skips in dynamic_max_skips_options:
                    for quantile in dynamic_quantiles:
                        thresholds = []
                        for position, values in enumerate(per_position_values):
                            if (
                                position == 0
                                or position == num_positions - 1
                                or position not in allowed_set
                                or not values
                            ):
                                thresholds.append(disabled_thresholds[position])
                                continue
                            tensor = torch.tensor(values, dtype=torch.float32)
                            thresholds.append(float(torch.quantile(tensor, quantile).item()))

                        metrics = run_eval(
                            return_routing_info=True,
                            enable_skipping=False,
                            return_batch_metrics=True,
                            seed=args.dynamic_skip_eval_seed,
                            dynamic_skip_strategy=args.dynamic_skip_strategy,
                            dynamic_skip_granularity=args.dynamic_skip_granularity,
                            dynamic_skip_probe_mode=probe_mode,
                            dynamic_skip_position_thresholds=thresholds,
                            dynamic_skip_max_skips=max_skips,
                        )
                        metrics["probe_mode"] = probe_mode
                        metrics["quantile"] = quantile
                        metrics["max_skips"] = max_skips
                        metrics["position_mode"] = mode_name
                        metrics["allowed_positions"] = sorted(allowed_set)
                        metrics["position_thresholds"] = thresholds
                        dynamic_results.append(metrics)

        best_dynamic_ppl = min(
            dynamic_results,
            key=lambda item: (
                item["perplexity"],
                -item.get("avg_compute_units", 0.0) if args.dynamic_skip_granularity == "mlp" else -item.get("avg_blocks", 0.0),
            ),
        )
        best_dynamic_skip = min(
            dynamic_results,
            key=lambda item: (
                dynamic_depth_key(item),
                item["perplexity"],
            ),
        )
        tolerated_candidates = [
            item
            for item in dynamic_results
            if item["perplexity"] <= dynamic_eval_full["perplexity"] * (1.0 + args.ppl_tolerance)
        ]
        best_dynamic_tolerated = min(
            tolerated_candidates,
            key=lambda item: (
                dynamic_depth_key(item),
                item["perplexity"],
            ),
            default=best_dynamic_ppl,
        )
        recommended_candidates = [item for item in dynamic_results if item.get("position_mode") == "recommended"]
        best_dynamic_recommended = min(
            recommended_candidates,
            key=lambda item: (
                item["perplexity"],
                -item.get("avg_compute_units", 0.0) if args.dynamic_skip_granularity == "mlp" else -item.get("avg_blocks", 0.0),
            ),
            default=best_dynamic_tolerated,
        )
        latency_baseline = None
        latency_candidates = []
        best_dynamic_speed = None
        if args.dynamic_skip_latency_num_batches > 0:
            speed_candidates = select_fast_latency_candidates(
                dynamic_results,
                full_perplexity=dynamic_eval_full["perplexity"],
                speed_ppl_tolerance=args.dynamic_skip_speed_ppl_tolerance,
                granularity=args.dynamic_skip_granularity,
                top_k=args.dynamic_skip_latency_top_k,
            )
            latency_baseline = run_latency_eval(
                seed=args.dynamic_skip_eval_seed,
                num_batches=args.dynamic_skip_latency_num_batches,
            )
            for item in speed_candidates:
                latency = run_latency_eval(
                    seed=args.dynamic_skip_eval_seed,
                    num_batches=args.dynamic_skip_latency_num_batches,
                    dynamic_skip_strategy=args.dynamic_skip_strategy,
                    dynamic_skip_granularity=args.dynamic_skip_granularity,
                    dynamic_skip_probe_mode=item.get("probe_mode", "all"),
                    dynamic_skip_position_thresholds=item["position_thresholds"],
                    dynamic_skip_max_skips=item["max_skips"],
                )
                candidate_payload = dict(item)
                candidate_payload["latency"] = latency
                candidate_payload["speedup_vs_full"] = latency_baseline["mean_batch_s"] / max(
                    latency["mean_batch_s"], 1e-8
                )
                latency_candidates.append(candidate_payload)
            if latency_candidates:
                best_dynamic_speed = min(
                    latency_candidates,
                    key=lambda item: (
                        item["latency"]["mean_batch_s"],
                        item["perplexity"],
                    ),
                )

        dynamic_skip_analysis = {
            "strategy": args.dynamic_skip_strategy,
            "granularity": args.dynamic_skip_granularity,
            "reference_probe_mode": reference_probe_mode,
            "metric_coverage": {
                probe_mode: probe_calibration["metric_coverage"]
                for probe_mode, probe_calibration in calibration_by_probe_mode.items()
            },
            "calibration_eval": reference_calibration,
            "calibration_by_probe_mode": {
                probe_mode: probe_calibration["calibration_eval"]
                for probe_mode, probe_calibration in calibration_by_probe_mode.items()
            },
            "eval_full": dynamic_eval_full,
            "results": dynamic_results,
            "tolerated_candidate_count": len(tolerated_candidates),
            "has_tolerated_candidate": bool(tolerated_candidates),
            "best_ppl_metrics": best_dynamic_ppl,
            "best_skip_metrics": best_dynamic_skip,
            "best_tolerated_metrics": best_dynamic_tolerated,
            "best_recommended_metrics": best_dynamic_recommended,
            "speed_ppl_tolerance": args.dynamic_skip_speed_ppl_tolerance,
            "latency_baseline": latency_baseline,
            "latency_candidates": latency_candidates,
            "best_speed_metrics": best_dynamic_speed,
        }

    best = min(
        [
            entry
            for entry in sweep
            if entry["perplexity"] <= full_eval["perplexity"] * (1.0 + args.ppl_tolerance)
        ],
        key=lambda entry: (entry.get("avg_blocks", float("inf")), entry["perplexity"]),
        default=sweep[0],
    )

    payload = {
        "model_path": args.model_path,
        "full_eval": full_eval,
        "skip_sweep": sweep,
        "percentile_sweep": percentile_sweep,
        "block_ablation": block_ablation,
        "difficulty_buckets": difficulty_buckets,
        "dynamic_skip_analysis": dynamic_skip_analysis,
        "best_threshold": best["threshold"],
        "best_keep_mask": best["keep_mask"],
        "best_metrics": best,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "routing_analysis.json", payload)

    if args.export_best_model_dir:
        export_dir = Path(args.export_best_model_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        decoder.set_skip_keep_mask(best["keep_mask"])
        model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)
        save_json(export_dir / "routing_analysis.json", payload)
        decoder.clear_skip_keep_mask()

    if args.export_best_dynamic_model_dir and dynamic_skip_analysis is not None:
        export_dir = Path(args.export_best_dynamic_model_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        best_dynamic_deploy = dynamic_skip_analysis["best_tolerated_metrics"]
        decoder.clear_skip_keep_mask()
        decoder.set_dynamic_skip_policy(
            strategy=args.dynamic_skip_strategy,
            granularity=args.dynamic_skip_granularity,
            position_thresholds=best_dynamic_deploy["position_thresholds"],
            max_skips=best_dynamic_deploy["max_skips"],
        )
        model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)
        save_json(export_dir / "routing_analysis.json", payload)
        decoder.clear_dynamic_skip_policy()

    print(f"Full-depth perplexity: {full_eval['perplexity']:.4f}")
    print(
        f"Best calibrated skip threshold: {best['threshold']:.4f} | "
        f"ppl={best['perplexity']:.4f} | avg_blocks={best.get('avg_blocks', 0):.2f}"
    )
    print(f"Best keep mask: {best['keep_mask']}")
    if dynamic_skip_analysis is not None:
        best_dynamic_ppl = dynamic_skip_analysis["best_ppl_metrics"]
        best_dynamic_skip = dynamic_skip_analysis["best_skip_metrics"]
        best_dynamic_tolerated = dynamic_skip_analysis["best_tolerated_metrics"]
        best_dynamic_recommended = dynamic_skip_analysis["best_recommended_metrics"]
        coverage = dynamic_skip_analysis.get("metric_coverage", {})
        if coverage and isinstance(next(iter(coverage.values())), dict):
            coverage_text = ", ".join(
                f"{probe}:pos={stats.get('covered_positions', 0)} values={stats.get('num_values', 0)}"
                for probe, stats in coverage.items()
            )
            print(f"Dynamic metric coverage: {coverage_text}")
        else:
            print(
                f"Dynamic metric coverage: positions={coverage.get('covered_positions', 0)} "
                f"values={coverage.get('num_values', 0)}"
            )
        print(
            f"Best dynamic skip quality: probe={best_dynamic_ppl['probe_mode']} mode={best_dynamic_ppl['position_mode']} "
            f"q={best_dynamic_ppl['quantile']:.3f} "
            f"max_skips={best_dynamic_ppl['max_skips']} "
            f"ppl={best_dynamic_ppl['perplexity']:.4f} "
            f"{'avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks'}="
            f"{best_dynamic_ppl.get('avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks', 0):.2f}"
        )
        print(
            f"Best dynamic skip tolerated: probe={best_dynamic_tolerated['probe_mode']} mode={best_dynamic_tolerated['position_mode']} "
            f"q={best_dynamic_tolerated['quantile']:.3f} "
            f"max_skips={best_dynamic_tolerated['max_skips']} "
            f"ppl={best_dynamic_tolerated['perplexity']:.4f} "
            f"{'avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks'}="
            f"{best_dynamic_tolerated.get('avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks', 0):.2f}"
        )
        if not dynamic_skip_analysis.get("has_tolerated_candidate", False):
            print(
                "Warning: no dynamic skip candidate satisfied ppl_tolerance; "
                "the tolerated export falls back to the best-quality dynamic configuration."
            )
        print(
            f"Best dynamic skip recommended: probe={best_dynamic_recommended['probe_mode']} mode={best_dynamic_recommended['position_mode']} "
            f"q={best_dynamic_recommended['quantile']:.3f} "
            f"max_skips={best_dynamic_recommended['max_skips']} "
            f"ppl={best_dynamic_recommended['perplexity']:.4f} "
            f"{'avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks'}="
            f"{best_dynamic_recommended.get('avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks', 0):.2f}"
        )
        print(
            f"Best dynamic skip depth: probe={best_dynamic_skip['probe_mode']} mode={best_dynamic_skip['position_mode']} "
            f"q={best_dynamic_skip['quantile']:.3f} "
            f"max_skips={best_dynamic_skip['max_skips']} "
            f"ppl={best_dynamic_skip['perplexity']:.4f} "
            f"{'avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks'}="
            f"{best_dynamic_skip.get('avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks', 0):.2f}"
        )
        if dynamic_skip_analysis.get("best_speed_metrics") is not None:
            best_dynamic_speed = dynamic_skip_analysis["best_speed_metrics"]
            latency = best_dynamic_speed["latency"]
            print(
                f"Best dynamic skip speed: probe={best_dynamic_speed['probe_mode']} mode={best_dynamic_speed['position_mode']} "
                f"q={best_dynamic_speed['quantile']:.3f} max_skips={best_dynamic_speed['max_skips']} "
                f"ppl={best_dynamic_speed['perplexity']:.4f} "
                f"{'avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks'}="
                f"{best_dynamic_speed.get('avg_compute_units' if args.dynamic_skip_granularity == 'mlp' else 'avg_blocks', 0):.2f} "
                f"mean_batch_s={latency['mean_batch_s']:.6f} speedup={best_dynamic_speed['speedup_vs_full']:.3f}x"
            )


if __name__ == "__main__":
    main()
