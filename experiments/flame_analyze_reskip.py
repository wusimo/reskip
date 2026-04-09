from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from flame_reskip_common import (
    evaluate_causal_lm,
    load_model_and_tokenizer,
    parse_csv_floats,
    save_json,
    build_text_dataloader,
)


def dynamic_metric_from_trace(entry: dict, strategy: str) -> float | None:
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
    if strategy == "recent_x_entropy_lt":
        if recent is None or entropy is None:
            return None
        return float(recent) * max(1.0 - float(entropy), 0.0)
    raise ValueError(f"Unsupported dynamic skip strategy: {strategy}")


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
    parser.add_argument("--dynamic_skip_quantiles", default="0.8,0.85,0.9,0.92,0.95,0.97,0.99")
    parser.add_argument("--dynamic_skip_max_skips_options", default="1,2")
    parser.add_argument("--dynamic_skip_calibration_batches", type=int, default=0)
    parser.add_argument("--dynamic_skip_calibration_seed", type=int, default=0)
    parser.add_argument("--dynamic_skip_eval_seed", type=int, default=1)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, dtype=args.dtype)

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
        calibration_batches = (
            args.dynamic_skip_calibration_batches
            if args.dynamic_skip_calibration_batches > 0
            else args.num_batches
        )
        dynamic_calibration = run_eval(
            return_routing_info=True,
            enable_skipping=False,
            return_batch_metrics=True,
            seed=args.dynamic_skip_calibration_seed,
            num_batches=calibration_batches,
        )
        num_positions = len(dynamic_calibration.get("block_importance", []))
        per_position_values: list[list[float]] = [[] for _ in range(num_positions)]
        for batch in dynamic_calibration.get("batch_metrics", []):
            for trace_entry in batch.get("execution_trace", []):
                position = int(trace_entry["position"])
                value = dynamic_metric_from_trace(trace_entry, args.dynamic_skip_strategy)
                if value is None:
                    continue
                per_position_values[position].append(float(value))

        dynamic_results = []
        dynamic_quantiles = parse_csv_floats(args.dynamic_skip_quantiles)
        dynamic_max_skips_options = [
            int(item.strip()) for item in args.dynamic_skip_max_skips_options.split(",") if item.strip()
        ]
        dynamic_eval_full = run_eval(
            return_routing_info=True,
            enable_skipping=False,
            return_batch_metrics=True,
            seed=args.dynamic_skip_eval_seed,
        )
        for max_skips in dynamic_max_skips_options:
            for quantile in dynamic_quantiles:
                thresholds = []
                for position, values in enumerate(per_position_values):
                    if position == 0 or position == num_positions - 1 or not values:
                        thresholds.append(1e9 if args.dynamic_skip_strategy.endswith("_gt") else -1e9)
                        continue
                    tensor = torch.tensor(values, dtype=torch.float32)
                    thresholds.append(float(torch.quantile(tensor, quantile).item()))

                metrics = run_eval(
                    return_routing_info=True,
                    enable_skipping=False,
                    return_batch_metrics=True,
                    seed=args.dynamic_skip_eval_seed,
                    dynamic_skip_strategy=args.dynamic_skip_strategy,
                    dynamic_skip_position_thresholds=thresholds,
                    dynamic_skip_max_skips=max_skips,
                )
                metrics["quantile"] = quantile
                metrics["max_skips"] = max_skips
                metrics["position_thresholds"] = thresholds
                dynamic_results.append(metrics)

        best_dynamic_ppl = min(
            dynamic_results,
            key=lambda item: (
                item["perplexity"],
                -item.get("avg_blocks", 0.0),
            ),
        )
        best_dynamic_skip = min(
            dynamic_results,
            key=lambda item: (
                item.get("avg_blocks", 0.0),
                item["perplexity"],
            ),
        )
        best_dynamic_tolerated = min(
            [
                item
                for item in dynamic_results
                if item["perplexity"] <= dynamic_eval_full["perplexity"] * (1.0 + args.ppl_tolerance)
            ],
            key=lambda item: (
                item.get("avg_blocks", float("inf")),
                item["perplexity"],
            ),
            default=best_dynamic_ppl,
        )
        dynamic_skip_analysis = {
            "strategy": args.dynamic_skip_strategy,
            "calibration_eval": dynamic_calibration,
            "eval_full": dynamic_eval_full,
            "results": dynamic_results,
            "best_ppl_metrics": best_dynamic_ppl,
            "best_skip_metrics": best_dynamic_skip,
            "best_tolerated_metrics": best_dynamic_tolerated,
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
        print(
            f"Best dynamic skip quality: q={best_dynamic_ppl['quantile']:.3f} "
            f"max_skips={best_dynamic_ppl['max_skips']} "
            f"ppl={best_dynamic_ppl['perplexity']:.4f} avg_blocks={best_dynamic_ppl.get('avg_blocks', 0):.2f}"
        )
        print(
            f"Best dynamic skip tolerated: q={best_dynamic_tolerated['quantile']:.3f} "
            f"max_skips={best_dynamic_tolerated['max_skips']} "
            f"ppl={best_dynamic_tolerated['perplexity']:.4f} avg_blocks={best_dynamic_tolerated.get('avg_blocks', 0):.2f}"
        )
        print(
            f"Best dynamic skip depth: q={best_dynamic_skip['quantile']:.3f} "
            f"max_skips={best_dynamic_skip['max_skips']} "
            f"ppl={best_dynamic_skip['perplexity']:.4f} avg_blocks={best_dynamic_skip.get('avg_blocks', 0):.2f}"
        )


if __name__ == "__main__":
    main()
