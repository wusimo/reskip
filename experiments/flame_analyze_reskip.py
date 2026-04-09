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
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, dtype=args.dtype)

    def run_eval(
        *,
        return_routing_info: bool,
        enable_skipping: bool | None,
        skip_keep_mask=None,
        return_batch_metrics: bool = False,
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
        )
        return evaluate_causal_lm(
            model,
            dataloader,
            args.device,
            num_batches=args.num_batches,
            return_routing_info=return_routing_info,
            enable_skipping=enable_skipping,
            skip_keep_mask=skip_keep_mask,
            return_batch_metrics=return_batch_metrics,
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

    print(f"Full-depth perplexity: {full_eval['perplexity']:.4f}")
    print(
        f"Best calibrated skip threshold: {best['threshold']:.4f} | "
        f"ppl={best['perplexity']:.4f} | avg_blocks={best.get('avg_blocks', 0):.2f}"
    )
    print(f"Best keep mask: {best['keep_mask']}")


if __name__ == "__main__":
    main()
