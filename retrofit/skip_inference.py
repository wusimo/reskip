"""Test actual block-skipping at inference using Route A's learned α.

For a trained Route A retrofit:
  - Identify low-importance blocks (I(n) < threshold)
  - At inference, SKIP those blocks (pass identity)
  - Measure PPL
  - Compare with:
    - No skip (baseline inference through all blocks)
    - Random skip (control)
"""
from __future__ import annotations
import argparse
import math
import sys
from copy import deepcopy

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flash-linear-attention")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flame")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/experiments")

import fla  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer
from flame_reskip_common import build_text_dataloader, count_valid_tokens
from retrofit_model import RetrofitModel


MODEL_PATH = "/home/user01/Minko/reskip2/reskip/flame/saves/transformer_test"
DATA_PATH = "/home/user01/Minko/datasets/fineweb_edu_100BT"


def load_retrofit(checkpoint_path, route, num_blocks=6, device="cuda:0", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=dtype
    ).to(device)
    model = RetrofitModel(base, num_blocks=num_blocks, route=route).to(device=device, dtype=dtype)
    if route == "A":
        model.gate_logits.data = model.gate_logits.data.to(dtype)
    ck = torch.load(checkpoint_path, map_location=device)
    model.router.load_state_dict(ck["router"])
    if route == "A" and "gate_logits" in ck:
        model.gate_logits.data.copy_(ck["gate_logits"].to(device=device, dtype=dtype))
    if "base_state_dict" in ck:
        model.base_model.load_state_dict(ck["base_state_dict"])
    model.eval()
    return model, tok


def patched_forward(model, input_ids, labels, attention_mask=None, skip_blocks=None, **kwargs):
    """Forward with specific blocks skipped (identity)."""
    skip_blocks = set(skip_blocks or [])
    embeds = model.embedding(input_ids)
    completed = [embeds]
    prev = embeds

    for n in range(model.num_blocks):
        if model.route == "A":
            beta_n = torch.sigmoid(model.gate_logits[n])
            if n == 0:
                block_input = embeds
            else:
                routed, _ = model.router.route(position=n + 1, completed_outputs=completed)
                block_input = (1 - beta_n) * prev + beta_n * routed
            if n in skip_blocks:
                # Skip this block's computation, pass through
                block_out = block_input
            else:
                block_out = model.block_forward(n, block_input, attention_mask=attention_mask, **kwargs)
            prev = block_out
            completed.append(block_out)
        else:
            # Route B: unchanged forward, skipping means identity on the block
            block_input = prev
            if n in skip_blocks:
                block_out = block_input
            else:
                block_out = model.block_forward(n, block_input, attention_mask=attention_mask, **kwargs)
            prev = block_out
            completed.append(block_out)

    hidden_states = model.final_norm(completed[-1])
    logits = model.lm_head(hidden_states)

    if labels is not None:
        shifted = torch.cat([labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model.vocab_size), shifted.view(-1), ignore_index=-100,
        )
        return loss
    return None


@torch.no_grad()
def measure_ppl_with_skip(model, tok, device, skip_blocks, num_batches=8, seq_len=8192):
    model.eval()
    dl = build_text_dataloader(
        tokenizer=tok, dataset=DATA_PATH, dataset_name=None, dataset_split="train",
        data_dir=None, data_files=None, seq_len=seq_len, context_len=2048,
        batch_size=1, num_workers=2, streaming=True, varlen=True, seed=0,
    )
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(dl):
        if i >= num_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        cu = batch.get("cu_seqlens")
        if cu is not None:
            cu = cu.to(device)
        kw = {"cu_seqlens": cu} if cu is not None else {}
        loss = patched_forward(model, ids, labels, skip_blocks=skip_blocks, **kw)
        n = count_valid_tokens(labels)
        total_loss += float(loss) * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--route", choices=["A", "B"], default="A")
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    model, tok = load_retrofit(args.checkpoint, args.route, args.num_blocks, device=device)

    # Baseline: no skip
    base_ppl = measure_ppl_with_skip(model, tok, device, skip_blocks=[], num_batches=args.num_batches)
    print(f"No skip (baseline): PPL={base_ppl:.3f}")

    # Skip each block individually
    print(f"\nSkip ONE block:")
    for b in range(args.num_blocks):
        ppl = measure_ppl_with_skip(model, tok, device, skip_blocks=[b], num_batches=args.num_batches)
        print(f"  skip block {b}: PPL={ppl:.3f} (Δ={ppl - base_ppl:+.3f})")

    # Skip combinations based on α importance (block 4 was 0)
    print(f"\nSkip multiple blocks:")
    for skip_set in [[4], [3, 4], [4, 5], [2, 4], [1, 3, 4], [0, 2, 4]]:
        ppl = measure_ppl_with_skip(model, tok, device, skip_blocks=skip_set, num_batches=args.num_batches)
        flops_pct = (args.num_blocks - len(skip_set)) / args.num_blocks * 100
        print(f"  skip blocks {skip_set}: PPL={ppl:.3f} (Δ={ppl - base_ppl:+.3f}, {flops_pct:.0f}% FLOPs)")


if __name__ == "__main__":
    main()
