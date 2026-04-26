"""Export models with ablation-informed dynamic skip policies for lm-eval."""
from __future__ import annotations

import argparse
import torch
from pathlib import Path

from flame_reskip_common import (
    build_text_dataloader,
    count_valid_tokens,
    load_model_and_tokenizer,
    save_json,
)


def dynamic_metric_from_trace(entry: dict) -> float | None:
    return entry.get("avg_phase1_recent_weight")


def dynamic_disable_threshold() -> float:
    return 1e9


@torch.no_grad()
def calibrate(model, dataloader, device, *, num_batches, num_positions, probe_mode):
    disabled = [dynamic_disable_threshold()] * num_positions
    per_pos: list[list[float]] = [[] for _ in range(num_positions)]
    n = 0
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        am = batch.get("attention_mask")
        cu = batch.get("cu_seqlens")
        if am is not None:
            am = am.to(device)
            if torch.all(am):
                am = None
        if cu is not None:
            cu = cu.to(device)
        out = model(
            input_ids=ids, attention_mask=am, labels=labels,
            cu_seqlens=cu, use_cache=False, return_dict=True,
            return_routing_info=True, enable_skipping=False,
            dynamic_skip_strategy="recent_weight_gt",
            dynamic_skip_granularity="block",
            dynamic_skip_probe_mode=probe_mode,
            dynamic_skip_position_thresholds=disabled,
            dynamic_skip_max_skips=0,
        )
        ri = getattr(out, "routing_info", None)
        if ri:
            for entry in ri.get("execution_trace", []):
                pos = int(entry["position"])
                if pos < num_positions:
                    val = dynamic_metric_from_trace(entry)
                    if val is not None:
                        per_pos[pos].append(float(val))
        n += 1
        if n >= num_batches:
            break
    return per_pos


def build_thresholds(per_pos, quantile, allowed, num_positions):
    dis = dynamic_disable_threshold()
    thresholds = []
    for pos in range(num_positions):
        values = per_pos[pos] if pos < len(per_pos) else []
        if pos == 0 or pos == num_positions - 1 or pos not in allowed or not values:
            thresholds.append(dis)
        else:
            t = torch.tensor(values, dtype=torch.float32)
            thresholds.append(float(torch.quantile(t, quantile).item()))
    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="flame/saves/reskip_transformer-340M")
    parser.add_argument("--dataset", default="/home/user01/Minko/datasets/fineweb_edu_100BT")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--cal_batches", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_base", default="outputs")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    decoder = model.model
    num_blocks = decoder.num_block_positions
    probe_mode = "attn_only"

    loader = build_text_dataloader(
        tokenizer=tokenizer, dataset=args.dataset, dataset_name=None,
        dataset_split="train", data_dir=None, data_files=None,
        seq_len=args.seq_len, context_len=args.seq_len,
        batch_size=1, num_workers=2, streaming=True, varlen=False, seed=0,
    )

    print("Calibrating...")
    per_pos = calibrate(model, loader, args.device,
                        num_batches=args.cal_batches,
                        num_positions=num_blocks,
                        probe_mode=probe_mode)

    configs = [
        ("ablation1_skip1_q085", {3}, 1, 0.85),
        ("ablation2_skip2_q085", {2, 3}, 2, 0.85),
        ("ablation2_skip2_q093", {2, 3}, 2, 0.93),
    ]

    for name, positions, max_skips, quantile in configs:
        thresholds = build_thresholds(per_pos, quantile, positions, num_blocks)
        out_dir = Path(args.output_base) / f"reskip_340M_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        decoder.clear_dynamic_skip_policy()
        decoder.set_dynamic_skip_policy(
            strategy="recent_weight_gt",
            probe_mode=probe_mode,
            position_thresholds=thresholds,
            max_skips=max_skips,
        )
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        save_json(out_dir / "skip_policy.json", {
            "strategy": "recent_weight_gt",
            "probe_mode": probe_mode,
            "positions": sorted(positions),
            "max_skips": max_skips,
            "quantile": quantile,
            "thresholds": thresholds,
        })
        print(f"Exported: {out_dir}")
        print(f"  positions={sorted(positions)} max_skips={max_skips} q={quantile}")
        print(f"  thresholds={[f'{t:.4f}' for t in thresholds]}")

    decoder.clear_dynamic_skip_policy()
    print("Done.")


if __name__ == "__main__":
    main()
