from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--thresholds", default="0.0,0.001,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--ppl_tolerance", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="outputs/flame_analysis")
    parser.add_argument("--export_best_model_dir", default="")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
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

    decoder = model.get_decoder()
    decoder.clear_skip_keep_mask()
    full_eval = evaluate_causal_lm(
        model,
        dataloader,
        args.device,
        num_batches=args.num_batches,
        return_routing_info=True,
        enable_skipping=False,
    )

    thresholds = parse_csv_floats(args.thresholds)
    sweep = []
    for threshold in thresholds:
        keep_mask = decoder.build_keep_mask_from_importance(full_eval["block_importance"], threshold)
        metrics = evaluate_causal_lm(
            model,
            dataloader,
            args.device,
            num_batches=args.num_batches,
            return_routing_info=True,
            enable_skipping=True,
            skip_keep_mask=keep_mask,
        )
        metrics["threshold"] = threshold
        metrics["keep_mask"] = keep_mask
        sweep.append(metrics)

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
