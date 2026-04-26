"""Wall-clock speed benchmark for Qwen3VLAttnResRetrofit.

Compares forward-pass latency at various seq lengths:
  - Base Qwen3-VL-2B (no retrofit)
  - Retrofit full-path
  - Retrofit static skip [blocks]
  - Retrofit dynamic skip (calibrated thresholds, Part 1 style)

Measurement: 5 warmup + 20 timed passes with torch.cuda.synchronize.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from collections import defaultdict

import torch
from datasets import load_dataset

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoTokenizer
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load(state_path, device):
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    if state_path is None:
        base.eval()
        return base, tok, None
    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kwargs = dict(num_blocks=cfg.get("num_blocks", 14))
    if "adapter_rank" in cfg: kwargs["adapter_rank"] = cfg["adapter_rank"]
    model = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype)
    model.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    model.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    model.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    model.eval()
    return model, tok, ck


@torch.no_grad()
def calibrate(model, tok, device, n_samples=32):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    ds = ds.select(range(n_samples, n_samples + 32))
    per_block = defaultdict(list)
    for ex in ds:
        text = ex["text"].strip()
        ids = tok.encode(text, add_special_tokens=False)[:512]
        inp = torch.tensor([ids], device=device)
        out = model(input_ids=inp, return_alpha=True)
        for i, tr in enumerate(out.skip_trace or []):
            w = tr.get("w_recent")
            if w is not None:
                per_block[i].append(w)
    return {b: sorted(v) for b, v in per_block.items()}


def pick_thr(per_block, q):
    return {b: vals[int(q * (len(vals) - 1))] for b, vals in per_block.items() if vals}


@torch.no_grad()
def bench_one(model, input_ids, n_warmup, n_timed, skip_kwargs):
    # Warmup
    for _ in range(n_warmup):
        _ = forward_once(model, input_ids, skip_kwargs)
        torch.cuda.synchronize()
    # Timed
    starts_ends = []
    for _ in range(n_timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = forward_once(model, input_ids, skip_kwargs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        starts_ends.append(t1 - t0)
    return starts_ends


def forward_once(model, input_ids, skip_kwargs):
    if isinstance(model, Qwen3VLAttnResRetrofit):
        return model(input_ids=input_ids, **skip_kwargs)
    else:
        return model(input_ids=input_ids, use_cache=False)


def report(name, times, seq_len):
    med = statistics.median(times) * 1000
    mean = statistics.mean(times) * 1000
    tok_s = seq_len / statistics.median(times)
    print(f"  {name:45s} median={med:7.2f}ms  mean={mean:7.2f}ms  tok/s={tok_s:8.0f}", flush=True)
    return med


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-path", required=True)
    p.add_argument("--seq-lens", default="512,1024,2048,4096")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    device = f"cuda:{args.gpu}"

    print(f"[speed] Loading base...")
    base_model, tok, _ = load(None, device)
    print(f"[speed] Loading retrofit...")
    retrofit, _, _ = load(args.state_path, device)

    print(f"[speed] Calibrating dynamic thresholds (q=0.85, M=2, P={{4,6}})...")
    per_block = calibrate(retrofit, tok, device)
    thr = pick_thr(per_block, q=0.85)
    dyn_cfg = dict(thresholds=thr, eligible_blocks={4, 6}, max_skips=2)

    for seq_len in [int(x) for x in args.seq_lens.split(",")]:
        # Use random token IDs within vocab
        input_ids = torch.randint(0, 100000, (1, seq_len), device=device)
        print(f"\n=== seq_len = {seq_len} ===", flush=True)

        base_times = bench_one(base_model, input_ids, args.warmup, args.timed, {})
        base_med = report("Base Qwen3-VL-2B", base_times, seq_len)

        retro_full = bench_one(retrofit, input_ids, args.warmup, args.timed, {})
        rf_med = report("Retrofit full-path", retro_full, seq_len)

        retro_static = bench_one(retrofit, input_ids, args.warmup, args.timed,
                                 {"skip_block_indices": [4]})
        rs_med = report("Retrofit static skip [4]", retro_static, seq_len)

        retro_dyn = bench_one(retrofit, input_ids, args.warmup, args.timed,
                              {"dynamic_skip_config": dyn_cfg})
        rd_med = report("Retrofit dynamic skip (q=0.85,M=2,P={4,6})", retro_dyn, seq_len)

        print(f"\n  Speedups vs base:")
        print(f"    retrofit full:   {base_med/rf_med:.3f}x")
        print(f"    static skip[4]:  {base_med/rs_med:.3f}x")
        print(f"    dynamic skip:    {base_med/rd_med:.3f}x")
        print(f"  Speedups vs retrofit-full:")
        print(f"    static skip[4]:  {rf_med/rs_med:.3f}x")
        print(f"    dynamic skip:    {rf_med/rd_med:.3f}x")


if __name__ == "__main__":
    main()
