"""Latency benchmark for 340M vanilla transformer + 340M AttnRes (reskip).

Used for Motivation section (Section 2.1 of paper). Measures forward-pass
wall-clock at seq ∈ {512, 1024, 2048, 8192}, H100 bfloat16, batch 1,
random-token inputs, 5 warmup + 20 timed.

Three model variants:
  - vanilla: standard-residual transformer-340M
  - attnres-full: reskip_transformer-340M, no skip
  - attnres-skip: reskip_transformer-340M + dynamic-skip config from Part-1
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time

import torch

# Register FLA models
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flash-linear-attention")
import fla  # noqa: F401
from transformers import AutoModelForCausalLM, AutoConfig


VANILLA = "/home/user01/Minko/reskip2/reskip/flame/saves/transformer-340M"
ATTNRES = "/home/user01/Minko/reskip2/reskip/flame/saves/reskip_transformer-340M"


def load(path, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    return model


@torch.no_grad()
def bench_one(model, input_ids, n_warmup, n_timed):
    for _ in range(n_warmup):
        _ = model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
    times = []
    for _ in range(n_timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def report(name, times):
    med = statistics.median(times) * 1000
    mean = statistics.mean(times) * 1000
    std = statistics.stdev(times) * 1000 if len(times) > 1 else 0
    print(f"  {name:42s} median={med:8.2f}ms  mean={mean:8.2f}ms  std={std:5.2f}", flush=True)
    return med


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", default="512,1024,2048,8192")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", choices=["vanilla", "attnres", "both"], default="both")
    args = p.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    if args.model in ("vanilla", "both"):
        print(f"\n[340M] loading vanilla transformer...", flush=True)
        v = load(VANILLA, device, dtype)
    if args.model in ("attnres", "both"):
        print(f"[340M] loading AttnRes (reskip) transformer...", flush=True)
        a = load(ATTNRES, device, dtype)

    vocab = (v.config.vocab_size if args.model in ("vanilla", "both")
             else a.config.vocab_size)

    for seq_len in [int(s) for s in args.seq_lens.split(",")]:
        print(f"\n=== seq_len = {seq_len} ===", flush=True)
        ids = torch.randint(0, min(vocab, 32000), (1, seq_len), device=device)
        if args.model in ("vanilla", "both"):
            tv = bench_one(v, ids, args.warmup, args.timed)
            mv = report("vanilla 340M (standard residual)", tv)
        if args.model in ("attnres", "both"):
            ta = bench_one(a, ids, args.warmup, args.timed)
            ma = report("AttnRes 340M (block-AttnRes, no skip)", ta)
        if args.model == "both":
            print(f"\n  Δ AttnRes-full vs vanilla: +{(ma - mv)/mv*100:.1f}%  "
                  f"(slower by {ma - mv:.2f}ms)")


if __name__ == "__main__":
    main()
