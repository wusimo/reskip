"""Integration benchmark: VLA in-backbone forward with real calibrated
thresholds, under use_cache ∈ {False, True} × skip ∈ {off, on}, at VLA-like
prefill lengths (1024, 2048) where vision tokens would sit.

Uses random token IDs so the run matches retrofit/bench_cache_regime.py's
measurement regime. This is the first benchmark that exercises the VLA
in-backbone skip path with a cache.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/starVLA/src")
from transformers import AutoModelForImageTextToText
from starvla_integration import StarVLAAttnResAdapter, StarVLABackboneSkipContext

MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"
STATE_PATH = "/home/user01/Minko/reskip2/reskip/retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt"


def load_adapter(base, device, dtype):
    cfg = base.config.text_config
    adapter = StarVLAAttnResAdapter(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        n_blocks=14,
        adapter_rank=256,
    ).to(device=device, dtype=dtype).eval()
    adapter.load_retrofit_state(STATE_PATH, strict=False)
    return adapter.to(device=device, dtype=dtype)


def load_dyn_cfg(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    thr = {int(k): float(v) for k, v in (d.get("thresholds") or {}).items()}
    return {
        "thresholds": thr,
        "eligible_blocks": set(d["eligible_blocks"]) if d.get("eligible_blocks") is not None else None,
        "max_skips": d.get("max_skips"),
    }


@torch.no_grad()
def time_fwd(base, adapter, ids, *, enable_skipping, dyn_cfg, use_cache, warmup, timed):
    ctx = StarVLABackboneSkipContext(adapter, enable_skipping=enable_skipping, dynamic_skip_config=dyn_cfg)
    ctx.bind_text_model(base.model.language_model)
    for _ in range(warmup):
        with ctx:
            _ = base(input_ids=ids, use_cache=use_cache)
        torch.cuda.synchronize()
    ts = []
    skipped_tally = []
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ctx:
            _ = base(input_ids=ids, use_cache=use_cache)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        skipped_tally.append(len(ctx.get_summary()["skipped_blocks"]))
    return statistics.median(ts) * 1000, statistics.mean(skipped_tally)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--seq-lens", default="1024,2048")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument("--dyn-cfg", default="/home/user01/Minko/reskip2/reskip/retrofit/outputs/vla_thresholds/h_r256_5k_q085.json")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    adapter = load_adapter(base, device, dtype)
    dyn_cfg = load_dyn_cfg(args.dyn_cfg)
    print(f"[bench] dyn_cfg loaded from {args.dyn_cfg}")
    print(f"        thresholds = {dyn_cfg['thresholds'] if dyn_cfg else None}")
    print(f"        eligible   = {sorted(dyn_cfg['eligible_blocks']) if dyn_cfg and dyn_cfg['eligible_blocks'] else None}")
    print(f"        max_skips  = {dyn_cfg.get('max_skips') if dyn_cfg else None}")

    for seq in [int(x) for x in args.seq_lens.split(",")]:
        ids = torch.randint(0, 100000, (1, seq), device=device)
        print(f"\n=== seq_len = {seq} ===")
        # Base comparator: stock base model (no adapter, no patch). Same ids.
        def bench_raw(use_cache):
            ts = []
            for _ in range(args.warmup):
                with torch.no_grad():
                    _ = base(input_ids=ids, use_cache=use_cache)
                torch.cuda.synchronize()
            for _ in range(args.timed):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = base(input_ids=ids, use_cache=use_cache)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            return statistics.median(ts) * 1000

        base_nc = bench_raw(False)
        base_c = bench_raw(True)
        rf_nc, _ = time_fwd(base, adapter, ids, enable_skipping=False, dyn_cfg=None, use_cache=False,
                            warmup=args.warmup, timed=args.timed)
        rf_c, _ = time_fwd(base, adapter, ids, enable_skipping=False, dyn_cfg=None, use_cache=True,
                           warmup=args.warmup, timed=args.timed)
        rs_nc, sk_nc = time_fwd(base, adapter, ids, enable_skipping=True, dyn_cfg=dyn_cfg, use_cache=False,
                                warmup=args.warmup, timed=args.timed)
        rs_c, sk_c = time_fwd(base, adapter, ids, enable_skipping=True, dyn_cfg=dyn_cfg, use_cache=True,
                              warmup=args.warmup, timed=args.timed)
        print(f"  {'config':40s} {'use_cache=F (ms)':>18s} {'use_cache=T (ms)':>18s}")
        print(f"  {'base (no AttnRes)':40s} {base_nc:>18.2f} {base_c:>18.2f}")
        print(f"  {'VLA in-backbone, full (no skip)':40s} {rf_nc:>18.2f} {rf_c:>18.2f}")
        print(f"  {'VLA in-backbone, dyn-skip':40s} {rs_nc:>18.2f} {rs_c:>18.2f}   "
              f"avg_skips={sk_nc:.2f}/{sk_c:.2f}")
        print(f"  ratio (dyn-skip, cache=T) vs base(cache=T): {rs_c/base_c:.3f}x")


if __name__ == "__main__":
    main()
