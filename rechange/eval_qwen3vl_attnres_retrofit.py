from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/rechange")
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=14)
    parser.add_argument("--adapter-rank", type=int, default=128)
    parser.add_argument("--attnres-blocks", default="", help="comma-separated block ids using AttnRes injection")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--prompt", default="Write one short sentence about adaptive computation.")
    parser.add_argument("--skip-blocks", default="10,11")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    reference = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    wrapped_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    attnres_blocks = [int(x) for x in args.attnres_blocks.split(",") if x.strip()]
    model = Qwen3VLAttnResRetrofit(
        wrapped_base,
        num_blocks=args.num_blocks,
        adapter_rank=args.adapter_rank,
        skippable_blocks=(attnres_blocks or None),
    ).to(device=device, dtype=dtype).eval()
    if args.state_path:
        ckpt = torch.load(args.state_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)

    inputs = processor(text=[args.prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    skip_blocks = [int(x) for x in args.skip_blocks.split(",") if x.strip()]

    with torch.no_grad():
        out_base = reference(**inputs, use_cache=False)
        out_full = model(**inputs, return_alpha=True, return_block_states=True)
        out_skip = model(
            **inputs,
            return_alpha=True,
            return_block_states=True,
            skip_block_indices=skip_blocks,
        )

    base_full = (out_base.logits - out_full.logits).abs().float()
    full_skip = (out_full.logits - out_skip.logits).abs().float()
    print(
        {
            "prompt": args.prompt,
            "skip_blocks": skip_blocks,
            "base_vs_full": {
                "max": float(base_full.max().item()),
                "mean": float(base_full.mean().item()),
            },
            "full_vs_skip": {
                "max": float(full_skip.max().item()),
                "mean": float(full_skip.mean().item()),
            },
            "actually_skipped": [
                item["block_idx"] for item in (out_skip.skip_trace or []) if item["skipped"]
            ],
            "gamma_mean": float(model.gamma.detach().float().mean().item()),
            "attnres_blocks": list(model.skippable_blocks),
        }
    )


if __name__ == "__main__":
    main()
