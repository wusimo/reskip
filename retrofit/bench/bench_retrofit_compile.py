"""Retrofit latency optimization: torch.compile vs eager.

Goal: close the 1.39× VLM-retrofit-vs-base gap at seq 2048 cache=T by
letting the PT2 compiler fuse router+adapter matmuls into base transformer
kernels. No retrain needed, pure inference-time win.

Reports:
  - TRUE base (eager vs compiled)
  - VLM retrofit (eager vs compiled)
  - ratio: retrofit-compiled / base-compiled  (goal: → 1.00)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit

MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"
DEFAULT_STATE_PATH = "/home/user01/Minko/reskip2/reskip/retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt"


@torch.no_grad()
def bench(call, warmup, timed):
    for _ in range(warmup):
        call()
        torch.cuda.synchronize()
    ts = []
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--seq-lens", default="1024,2048")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=15)
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    help="torch.compile mode: default, reduce-overhead, max-autotune")
    ap.add_argument("--state-path", default=DEFAULT_STATE_PATH,
                    help="retrofit_attnres_state.pt path (defaults to H_r256_5k 14-block)")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[compile-bench] loading models on {device} (mode={args.compile_mode})")
    print(f"[compile-bench] state: {args.state_path}")

    true_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    vlm_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    ck = torch.load(args.state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kw = dict(num_blocks=cfg.get("num_blocks", 14))
    if "adapter_rank" in cfg:
        kw["adapter_rank"] = cfg["adapter_rank"]
    vlm_retrofit = Qwen3VLAttnResRetrofit(vlm_base, **kw).to(device=device, dtype=dtype).eval()
    vlm_retrofit.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    vlm_retrofit.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    vlm_retrofit.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))

    # Freeze γ to 1.0 for inference (matches trained γ saturation).
    # This enables a potential fast-path where adapter(delta) is uncorrected.
    print(f"[compile-bench] γ values (pre-load): {vlm_retrofit.gamma.detach().cpu().tolist()}")

    # Compile only the language-model forward (where retrofit hooks).
    # Using reduce-overhead which enables CUDA graphs (needs stable input shape).
    base_compiled = torch.compile(true_base, mode=args.compile_mode, dynamic=False)
    retrofit_compiled = torch.compile(vlm_retrofit, mode=args.compile_mode, dynamic=False)

    for seq in [int(x) for x in args.seq_lens.split(",")]:
        ids = torch.randint(0, 100000, (1, seq), device=device)
        print(f"\n=== seq_len = {seq} (cache=T, bf16) ===")
        print(f"  {'config':40s} {'eager (ms)':>12s} {'compiled (ms)':>14s} {'speedup':>10s}")

        b_eager = bench(lambda: true_base(input_ids=ids, use_cache=True), args.warmup, args.timed)
        b_comp = bench(lambda: base_compiled(input_ids=ids, use_cache=True), args.warmup, args.timed)
        print(f"  {'TRUE base':40s} {b_eager:>12.2f} {b_comp:>14.2f} {b_eager/b_comp:>10.3f}x")

        v_eager = bench(lambda: vlm_retrofit(input_ids=ids, use_cache=True), args.warmup, args.timed)
        v_comp = bench(lambda: retrofit_compiled(input_ids=ids, use_cache=True), args.warmup, args.timed)
        print(f"  {'VLM retrofit':40s} {v_eager:>12.2f} {v_comp:>14.2f} {v_eager/v_comp:>10.3f}x")

        print(f"\n  ratios:")
        print(f"    retrofit_eager    / base_eager    = {v_eager/b_eager:.3f}x  (current)")
        print(f"    retrofit_compiled / base_compiled = {v_comp/b_comp:.3f}x  (compiled)")
        print(f"    retrofit_compiled / base_eager    = {v_comp/b_eager:.3f}x  (mixed)")


if __name__ == "__main__":
    main()
