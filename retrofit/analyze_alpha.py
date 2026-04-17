"""Analyze α routing patterns from a retrofitted model."""
from __future__ import annotations
import argparse
import math
import sys

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flash-linear-attention")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flame")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/experiments")

import fla  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer
from flame_reskip_common import build_text_dataloader
from retrofit_model import RetrofitModel


MODEL_PATH = "/home/user01/Minko/reskip2/reskip/flame/saves/transformer_test"
DATA_PATH = "/home/user01/Minko/datasets/fineweb_edu_100BT"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--route", choices=["A", "B"], required=True)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--num-batches", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=dtype
    ).to(device)

    model = RetrofitModel(base, num_blocks=args.num_blocks, route=args.route).to(device=device, dtype=dtype)
    if args.route == "A":
        model.gate_logits.data = model.gate_logits.data.to(dtype)

    ck = torch.load(args.checkpoint, map_location=device)
    model.router.load_state_dict(ck["router"])
    if args.route == "A" and "gate_logits" in ck:
        model.gate_logits.data.copy_(ck["gate_logits"].to(device=device, dtype=dtype))
    if "base_state_dict" in ck:
        model.base_model.load_state_dict(ck["base_state_dict"])
    model.eval()

    dl = build_text_dataloader(
        tokenizer=tok, dataset=DATA_PATH, dataset_name=None, dataset_split="train",
        data_dir=None, data_files=None, seq_len=8192, context_len=2048,
        batch_size=1, num_workers=2, streaming=True, varlen=True, seed=123,
    )

    # Accumulate mean α per position over batches
    # α[n] has shape [B, T, n+1] where n+1 sources are {embedding, block_0, ..., block_{n-1}}
    # For each block position n (1..num_blocks-1 in terms of completed sources),
    # mean over batch/seq → one distribution per n.
    per_position_alpha = {}  # key: position n (how many sources), value: [sources]
    for i, batch in enumerate(dl):
        if i >= args.num_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        cu = batch.get("cu_seqlens")
        if cu is not None:
            cu = cu.to(device)
        with torch.no_grad():
            out = model(input_ids=ids, labels=labels, cu_seqlens=cu, return_alpha=True)
        if out.alpha_list is None:
            continue
        for n, alpha in enumerate(out.alpha_list):
            # alpha: [B, T, num_sources]
            mean = alpha.mean(dim=(0, 1)).cpu().float().tolist()  # [num_sources]
            num_sources = len(mean)
            per_position_alpha.setdefault(num_sources, []).append(mean)

    print(f"{'=' * 60}")
    print(f"Route {args.route} — α distribution per block position")
    print(f"{'=' * 60}")
    print(f"{'n_src':>6s} {'alpha over sources (embed, block_0, ...)':<s}")
    for num_src in sorted(per_position_alpha.keys()):
        alphas = per_position_alpha[num_src]
        # Average across batches
        avg = [sum(a[i] for a in alphas) / len(alphas) for i in range(num_src)]
        argmax_src = max(range(num_src), key=lambda i: avg[i])
        fmt = " ".join(f"{v:.3f}" for v in avg)
        src_names = ["emb"] + [f"b{i}" for i in range(num_src - 1)]
        label_fmt = " ".join(f"{s:>5s}" for s in src_names)
        # Normalize the α format
        value_fmt = " ".join(f"{v:5.3f}" for v in avg)
        argmax_name = src_names[argmax_src]
        print(f"\n pos n={num_src} (consumes {num_src} sources):")
        print(f"   labels: {label_fmt}")
        print(f"   alphas: {value_fmt}")
        print(f"   argmax: {argmax_name} (α={avg[argmax_src]:.3f})")

    # Block importance: I(n) = max_{l > n} alpha_{n→l}
    print(f"\n{'=' * 60}")
    print("Block importance I(n) = max over downstream blocks' α at n")
    print(f"{'=' * 60}")
    num_blocks = args.num_blocks
    # Build full α matrix: M[src][dst] = avg α from source src when consumed by dst
    # src ∈ [0..num_blocks]: 0 = embedding, i = block_{i-1} output
    # dst ∈ [1..num_blocks]: block position that's being computed
    # α_list indexed by block position n=0..num_blocks-1; when n>=1, has (n+1) sources
    matrix = [[0.0] * (num_blocks + 1) for _ in range(num_blocks + 1)]
    for num_src in sorted(per_position_alpha.keys()):
        alphas = per_position_alpha[num_src]
        avg = [sum(a[i] for a in alphas) / len(alphas) for i in range(num_src)]
        dst = num_src  # dst = number of sources at this position
        for src in range(num_src):
            matrix[src][dst] = avg[src]

    # For each source (block idx 0..num_blocks-1 plus embed), print max downstream alpha
    for src_idx in range(num_blocks + 1):
        downstream = [matrix[src_idx][d] for d in range(src_idx + 1, num_blocks + 1)]
        if downstream:
            label = "embed" if src_idx == 0 else f"block {src_idx - 1}"
            print(f"  {label:>12s}: max downstream α = {max(downstream):.3f}")


if __name__ == "__main__":
    main()
