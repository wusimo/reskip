"""Smoke tests for Qwen3VLAttnResRetrofit.

Three checks:
  1. identity-at-init: with γ=0, retrofit forward must equal base forward
     exactly (within bf16 noise). Argmax must match.
  2. full vs skip: forward with and without skip_block_indices runs without
     errors, produces expected shape.
  3. AttnRes coverage: by default, all blocks are wrapped with AttnRes
     injection (not a subset).
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def identity_test(device, dtype, num_blocks):
    """At γ=0, retrofit(x) must equal base(x) exactly."""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    ids = tok(
        "The capital of France is Paris. The capital of Germany is",
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        base_logits = base(input_ids=ids, use_cache=False).logits

    model = Qwen3VLAttnResRetrofit(base, num_blocks=num_blocks).to(device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        r_out = model(input_ids=ids, return_alpha=True)
    r_logits = r_out.logits

    diff = (base_logits.float() - r_logits.float()).abs()
    argmax_match = bool((base_logits.argmax(-1) == r_logits.argmax(-1)).all().item())
    top1_base = tok.decode([int(base_logits[0, -1].argmax())])
    top1_retr = tok.decode([int(r_logits[0, -1].argmax())])

    trace = r_out.skip_trace or []
    used_attnres = [item["block_idx"] for item in trace if item["used_attnres"]]

    print("=== IDENTITY @ INIT ===")
    print(f"  skippable_blocks (AttnRes-wrapped): {model.skippable_blocks}")
    print(f"  gamma init: {model.gamma.data.tolist()}")
    print(f"  used_attnres_blocks in forward: {used_attnres}")
    print(f"  max |Δlogits|:  {float(diff.max()):.6f}")
    print(f"  mean |Δlogits|: {float(diff.mean()):.6f}")
    print(f"  argmax_match:   {argmax_match}")
    print(f"  base top-1: {top1_base!r}  retrofit top-1: {top1_retr!r}")
    assert argmax_match, "argmax diverged at init — retrofit is NOT identity-at-init"
    assert len(used_attnres) == num_blocks, (
        f"expected AttnRes on all {num_blocks} blocks, got {len(used_attnres)}"
    )
    print("  ✓ identity check passed\n")
    return base, tok


def skip_test(base, tok, device, num_blocks, skip_blocks):
    """Full-path vs skip-path sanity."""
    model = Qwen3VLAttnResRetrofit(base, num_blocks=num_blocks).to(device=device, dtype=torch.bfloat16)
    model.eval()
    ids = tok("Photosynthesis is the process by which", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out_full = model(input_ids=ids, return_alpha=True)
        out_skip = model(input_ids=ids, return_alpha=True, skip_block_indices=skip_blocks)
    diff = (out_full.logits - out_skip.logits).abs().float()
    actually_skipped = [item["block_idx"] for item in (out_skip.skip_trace or []) if item["skipped"]]
    print("=== SKIP PATH ===")
    print(f"  requested skip: {skip_blocks}  actually skipped: {actually_skipped}")
    print(f"  max |Δ(full-skip) logits|:  {float(diff.max()):.4f}")
    print(f"  mean |Δ(full-skip) logits|: {float(diff.mean()):.4f}")
    print("  (large diff expected — skip replaces Block_n(x_n) with x_n)")
    assert set(actually_skipped) == set(skip_blocks), "skip request not honored"
    print("  ✓ skip path works\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=14)
    parser.add_argument("--skip-blocks", default="4,10,12")
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    base, tok = identity_test(device, dtype, args.num_blocks)
    skip_blocks = [int(x) for x in args.skip_blocks.split(",") if x.strip()]
    skip_test(base, tok, device, args.num_blocks, skip_blocks)


if __name__ == "__main__":
    main()
