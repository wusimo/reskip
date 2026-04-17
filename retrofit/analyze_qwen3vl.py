"""Analyze α routing patterns from retrofitted Qwen3-VL-2B."""
from __future__ import annotations
import argparse
import math
import sys

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoTokenizer

from qwen3vl_retrofit import Qwen3VLRetrofit
from train_qwen3vl import build_text_stream


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load_retrofit(ckpt_path, num_blocks, device="cuda:0", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    model = Qwen3VLRetrofit(base, num_blocks=num_blocks, route="A").to(device=device, dtype=dtype)
    model.gate_logits.data = model.gate_logits.data.to(dtype)
    ck = torch.load(ckpt_path, map_location=device)
    model.router.load_state_dict(ck["router"])
    if "gate_logits" in ck:
        model.gate_logits.data.copy_(ck["gate_logits"].to(device=device, dtype=dtype))
    model.eval()
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-blocks", type=int, required=True)
    p.add_argument("--num-seqs", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    model, tok = load_retrofit(args.checkpoint, args.num_blocks, device=device)

    # Accumulate α per position over fineweb sequences
    per_pos_alpha = {}
    with torch.no_grad():
        stream = build_text_stream(tok, seq_len=2048, seed=777)
        for i, seq in enumerate(stream):
            if i >= args.num_seqs:
                break
            ids = seq.unsqueeze(0).to(device)
            out = model(input_ids=ids, labels=ids, return_alpha=True)
            for alpha in out.alpha_list:
                num_src = alpha.shape[-1]
                mean = alpha.float().mean(dim=(0, 1)).cpu().tolist()
                per_pos_alpha.setdefault(num_src, []).append(mean)

    print(f"{'='*80}")
    print(f"Qwen3-VL-2B α distribution per block position (num_blocks={args.num_blocks})")
    print(f"{'='*80}")
    num_blocks = args.num_blocks
    # Collect for importance matrix
    matrix = [[0.0] * (num_blocks + 1) for _ in range(num_blocks + 1)]
    for num_src in sorted(per_pos_alpha.keys()):
        alphas = per_pos_alpha[num_src]
        avg = [sum(a[i] for a in alphas) / len(alphas) for i in range(num_src)]
        dst = num_src  # number of completed sources when this α applied
        for src in range(num_src):
            matrix[src][dst] = avg[src]
        argmax = max(range(num_src), key=lambda i: avg[i])
        src_name = "emb" if argmax == 0 else f"b{argmax - 1}"
        top3 = sorted(range(num_src), key=lambda i: -avg[i])[:3]
        top3_str = ", ".join(
            f"{'emb' if i == 0 else f'b{i - 1}'}={avg[i]:.3f}" for i in top3
        )
        print(f"pos n={num_src:2d}: argmax={src_name} (α={avg[argmax]:.3f})  top3: {top3_str}")

    # Block importance
    print(f"\n{'='*80}")
    print("Block importance I(n) = max over downstream blocks' α at source n")
    print(f"{'='*80}")
    importances = []
    for src_idx in range(num_blocks + 1):
        downstream = [matrix[src_idx][d] for d in range(src_idx + 1, num_blocks + 1)]
        if downstream:
            imp = max(downstream)
            label = "embed" if src_idx == 0 else f"block {src_idx - 1}"
            mark = ""
            if src_idx > 0 and imp < 0.1:
                mark = " ← SKIPPABLE (I < 0.1)"
            elif src_idx > 0 and imp < 0.3:
                mark = " ← low importance"
            print(f"  {label:>12s}: I = {imp:.4f}{mark}")
            if src_idx > 0:  # block (not embed)
                importances.append((src_idx - 1, imp))

    # Rank blocks by importance
    sorted_blocks = sorted(importances, key=lambda x: x[1])
    print(f"\nLowest-importance blocks (candidates to skip):")
    for idx, (b, imp) in enumerate(sorted_blocks[:6]):
        print(f"  #{idx+1}: block {b} (I={imp:.4f})")


if __name__ == "__main__":
    main()
