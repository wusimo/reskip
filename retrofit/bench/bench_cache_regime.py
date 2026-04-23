"""Measure base vs retrofit latency under use_cache=True at realistic lengths.

Splits the cost into:
  - prefill: single forward over prompt tokens, cache populated
  - decode:  N further steps of 1-token forward reading + extending cache

Also covers the *skip+cache* path (retrofit + dyn-skip, use_cache=True): at
decode, skipped blocks write K/V via the K/V-only path so subsequent steps
have a consistent cache.

Prints median of 20 trials per cell (after warmup). Uses random token IDs so
we can test any prefill length without tokenizing real text.
"""
from __future__ import annotations

import argparse
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
    """Load a fresh UNPATCHED base and a separately-wrapped retrofit.

    BUG fixed 2026-04-20: the previous version returned the same ``base``
    that ``Qwen3VLAttnResRetrofit.__init__`` had monkey-patched. Benchmarks
    called the returned "base" and unintentionally measured retrofit-full.
    """
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    true_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    if state_path is None:
        return true_base, None, tok
    # Separate base for the retrofit wrapper so ``true_base.model.language_model.forward``
    # stays as HF's stock forward.
    retro_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kwargs = dict(num_blocks=cfg.get("num_blocks", 14))
    if "adapter_rank" in cfg:
        kwargs["adapter_rank"] = cfg["adapter_rank"]
    model = Qwen3VLAttnResRetrofit(retro_base, **kwargs).to(device=device, dtype=dtype).eval()
    model.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    model.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    model.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    return true_base, model, tok


@torch.no_grad()
def calibrate_thresholds(model, tok, device, n=32, seq_len=512):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test").select(range(n, n + n))
    per_block = defaultdict(list)
    for ex in ds:
        ids = tok.encode(ex["text"].strip(), add_special_tokens=False)[:seq_len]
        out = model(input_ids=torch.tensor([ids], device=device), return_alpha=True)
        for i, tr in enumerate(out.skip_trace or []):
            if tr.get("w_recent") is not None:
                per_block[i].append(tr["w_recent"])
    thr = {}
    for b, vals in per_block.items():
        if vals:
            vs = sorted(vals)
            thr[b] = vs[int(0.85 * (len(vs) - 1))]
    return thr


@torch.no_grad()
def generate_cached(m, input_ids, n_decode):
    """Manual prefill + decode loop. Returns (prefill_ms, decode_ms)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = m(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    t_pre = time.perf_counter() - t0

    pkv = out.past_key_values
    cur = out.logits[:, -1:].argmax(-1)  # [B, 1]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(n_decode):
        out = m(input_ids=cur, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize()
    t_dec = time.perf_counter() - t1
    return t_pre, t_dec


def bench(m, input_ids, n_decode, warmup, timed):
    for _ in range(warmup):
        generate_cached(m, input_ids, n_decode)
    prefill_times, decode_times = [], []
    for _ in range(timed):
        tp, td = generate_cached(m, input_ids, n_decode)
        prefill_times.append(tp)
        decode_times.append(td)
    return statistics.median(prefill_times) * 1000, statistics.median(decode_times) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-path", required=True)
    p.add_argument("--prefill-lens", default="1024,2048")
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timed", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--eligible", default="4,6")
    p.add_argument("--max-skips", type=int, default=2)
    args = p.parse_args()
    device = f"cuda:{args.gpu}"

    base_wrap, retro, tok = load(args.state_path, device)
    thr = calibrate_thresholds(retro, tok, device)
    eligible = set(int(x) for x in args.eligible.split(","))
    dyn_cfg = dict(thresholds=thr, eligible_blocks=eligible, max_skips=args.max_skips)
    print(f"decode tokens per call: {args.decode_tokens}, warmup={args.warmup}, timed={args.timed}", flush=True)
    print(f"dyn-skip config: q=0.85, max_skips={args.max_skips}, eligible={sorted(eligible)}")
    for seq in [int(x) for x in args.prefill_lens.split(",")]:
        ids = torch.randint(0, 100000, (1, seq), device=device)

        # 1) base, cache on
        base_pre, base_dec = bench(base_wrap, ids, args.decode_tokens, args.warmup, args.timed)
        # 2) retrofit full-path, cache on (no skip)
        retro._active_skip_blocks = set()
        retro._dynamic_skip_config = None
        rf_pre, rf_dec = bench(retro.base_model, ids, args.decode_tokens, args.warmup, args.timed)
        # 3) retrofit + dyn-skip, cache on
        retro._dynamic_skip_config = dyn_cfg
        rs_pre, rs_dec = bench(retro.base_model, ids, args.decode_tokens, args.warmup, args.timed)

        base_dec_per = base_dec / args.decode_tokens
        rf_dec_per = rf_dec / args.decode_tokens
        rs_dec_per = rs_dec / args.decode_tokens
        print(f"\n=== prefill seq_len = {seq}, decode = {args.decode_tokens} tok ===")
        print(f"  {'config':45s} {'prefill (ms)':>14s} {'decode total (ms)':>20s} {'decode/tok (ms)':>18s}")
        print(f"  {'Base Qwen3-VL-2B':45s} {base_pre:>14.2f} {base_dec:>20.2f} {base_dec_per:>18.3f}")
        print(f"  {'Retrofit full (trained γ=1)':45s} {rf_pre:>14.2f} {rf_dec:>20.2f} {rf_dec_per:>18.3f}")
        print(f"  {'Retrofit + dyn-skip':45s} {rs_pre:>14.2f} {rs_dec:>20.2f} {rs_dec_per:>18.3f}")
        print(f"  ratios vs base:")
        print(f"    retrofit full:   prefill={rf_pre/base_pre:.3f}x  decode/tok={rf_dec_per/base_dec_per:.3f}x")
        print(f"    retrofit+skip:   prefill={rs_pre/base_pre:.3f}x  decode/tok={rs_dec_per/base_dec_per:.3f}x")


if __name__ == "__main__":
    main()
