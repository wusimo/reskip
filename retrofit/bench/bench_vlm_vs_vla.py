"""Head-to-head: TRUE base vs VLM retrofit vs VLA in-backbone.

All three share the same H_r256_5k router/adapter/γ weights and run on the
same random-token inputs on the same GPU. Goal: prove that VLM-retrofit
and VLA-in-backbone carry identical router+adapter overhead (they should,
since they're ports of the same per-block forward), so any "VLM is faster
than VLA" gap observed earlier was a benchmark artifact, not a code
difference.

The TRUE base is loaded as a fresh instance with no retrofit wrapper ever
touching it, so base.model.language_model.forward is stock HF.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/starVLA/src")
from transformers import AutoModelForImageTextToText
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit
from starvla_integration import StarVLAAttnResAdapter, StarVLABackboneSkipContext

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
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--seq-lens", default="1024,2048")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument("--state-path", default=DEFAULT_STATE_PATH,
                    help="retrofit_attnres_state.pt path (defaults to H_r256_5k 14-block)")
    ap.add_argument("--vla-n-blocks", type=int, default=14,
                    help="num_blocks for the VLA-in-backbone adapter (must match state-path config)")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    # TRUE base — completely untouched by any wrapper.
    true_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()

    # VLM retrofit on its OWN base (so true_base stays stock).
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

    # VLA in-backbone on ANOTHER base.
    vla_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    hidden_size = vla_base.config.text_config.hidden_size
    num_hidden_layers = vla_base.config.text_config.num_hidden_layers
    vla_adapter = StarVLAAttnResAdapter(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        n_blocks=args.vla_n_blocks,
        adapter_rank=256,
    ).to(device=device, dtype=dtype).eval()
    vla_adapter.load_retrofit_state(args.state_path, strict=False)
    vla_adapter = vla_adapter.to(device=device, dtype=dtype)

    for seq in [int(x) for x in args.seq_lens.split(",")]:
        ids = torch.randint(0, 100000, (1, seq), device=device)
        print(f"\n=== seq_len = {seq} ===")
        print(f"  {'config':40s} {'cache=F (ms)':>14s} {'cache=T (ms)':>14s}")

        # 1) TRUE base
        b_nc = bench(lambda: true_base(input_ids=ids, use_cache=False), args.warmup, args.timed)
        b_c = bench(lambda: true_base(input_ids=ids, use_cache=True), args.warmup, args.timed)
        print(f"  {'(1) TRUE base (stock Qwen3-VL-2B)':40s} {b_nc:>14.2f} {b_c:>14.2f}")

        # 2) VLM retrofit full-path
        v_nc = bench(lambda: vlm_retrofit(input_ids=ids, use_cache=False), args.warmup, args.timed)
        v_c = bench(lambda: vlm_retrofit(input_ids=ids, use_cache=True), args.warmup, args.timed)
        print(f"  {'(2) VLM retrofit full (γ=1)':40s} {v_nc:>14.2f} {v_c:>14.2f}")

        # 3) VLA in-backbone full-path (no skip)
        ctx = StarVLABackboneSkipContext(vla_adapter, enable_skipping=False, dynamic_skip_config=None)
        ctx.bind_text_model(vla_base.model.language_model)
        def vla_full(uc):
            def _call():
                with ctx:
                    vla_base(input_ids=ids, use_cache=uc)
            return _call
        l_nc = bench(vla_full(False), args.warmup, args.timed)
        l_c = bench(vla_full(True), args.warmup, args.timed)
        print(f"  {'(3) VLA in-backbone full (γ=1)':40s} {l_nc:>14.2f} {l_c:>14.2f}")

        print(f"  ratios vs TRUE base under cache=T:")
        print(f"    (2) VLM retrofit:    {v_c/b_c:.3f}x")
        print(f"    (3) VLA in-backbone: {l_c/b_c:.3f}x")
        print(f"  VLM vs VLA: {v_c/l_c:.3f}x  (should be ≈ 1.00 if they're doing the same work)")


if __name__ == "__main__":
    main()
