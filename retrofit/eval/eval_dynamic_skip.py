"""Dynamic per-input skip eval for Qwen3VLAttnResRetrofit.

Matches Part 1 ReSkip's `recent_weight_gt` strategy:
  At block n, read phase-1 α and compute w_recent(n) = α[..., -1] mean
  If w_recent(n) > τ_n  AND  n ∈ P  AND  total_skips < M_max → skip block n

Pipeline:
  1. Calibrate: run the model on a small held-out set, collect w_recent per block
  2. Set τ_n = empirical quantile q (default 0.85) per block
  3. Eval with dynamic_skip_config on LAMBADA/HellaSwag, sweep (P, q, M_max)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoTokenizer
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load_trained(state_path, device, num_blocks=14):
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kwargs = dict(num_blocks=cfg.get("num_blocks", num_blocks))
    if "adapter_rank" in cfg: kwargs["adapter_rank"] = cfg["adapter_rank"]
    model = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype)
    model.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    model.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    model.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    model.eval()
    return model, tok


@torch.no_grad()
def calibrate_thresholds(model, tok, device, n_samples=32, seq_len=512):
    """Run on held-out LAMBADA prefixes, collect w_recent(n) per block.
    Returns per-block list of samples."""
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    ds = ds.select(range(n_samples, n_samples + 32))  # offset to avoid eval overlap
    per_block_w_recent = defaultdict(list)
    for ex in ds:
        text = ex["text"].strip()
        ids = tok.encode(text, add_special_tokens=False)[:seq_len]
        inp = torch.tensor([ids], device=device)
        out = model(input_ids=inp, return_alpha=True)
        for block_idx, trace in enumerate(out.skip_trace or []):
            w = trace.get("w_recent")
            if w is not None:
                per_block_w_recent[block_idx].append(w)
    return dict(per_block_w_recent)


def pick_thresholds(per_block_samples, q=0.85):
    """Pick τ_n as q-th quantile of w_recent(n) samples."""
    thresholds = {}
    for b, vals in per_block_samples.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        idx = int(q * (len(vals_sorted) - 1))
        thresholds[b] = vals_sorted[idx]
    return thresholds


@torch.no_grad()
def eval_dynamic_lambada(model, tok, device, eligible, thresholds, max_skips, n=500):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if n < len(ds):
        ds = ds.select(range(n))
    cfg = dict(thresholds=thresholds, eligible_blocks=set(eligible), max_skips=max_skips)
    correct = total = 0
    nll = 0.0; ntok = 0
    skip_counts = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        text = ex["text"].strip()
        if " " not in text: continue
        last = text.rfind(" ")
        ctx = text[:last]
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        full_ids = tok.encode(text, add_special_tokens=False)
        tgt_ids = full_ids[len(ctx_ids):]
        if not tgt_ids: continue
        inp = torch.tensor([ctx_ids + tgt_ids], device=device)
        out = model(input_ids=inp, dynamic_skip_config=cfg)
        logits = out.logits
        start = len(ctx_ids) - 1
        pred_logits = logits[0, start:start + len(tgt_ids), :]
        tgt = torch.tensor(tgt_ids, device=device)
        is_corr = bool((pred_logits.argmax(-1) == tgt).all().item())
        lp = F.log_softmax(pred_logits.float(), dim=-1)
        nll += -lp.gather(1, tgt.unsqueeze(-1)).sum().item()
        ntok += len(tgt_ids)
        correct += int(is_corr); total += 1
        if out.skip_trace:
            skip_counts.append(sum(1 for t in out.skip_trace if t["skipped"]))
        if (i + 1) % 100 == 0:
            avg_sk = sum(skip_counts) / max(len(skip_counts), 1)
            print(f"  {i+1}/{len(ds)} acc={correct/total:.4f} ppl={math.exp(nll/ntok):.2f} "
                  f"avg_skips={avg_sk:.2f} ({time.time()-t0:.0f}s)", flush=True)
    avg_sk = sum(skip_counts) / max(len(skip_counts), 1)
    return correct / max(total, 1), math.exp(nll / max(ntok, 1)), avg_sk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-path", required=True)
    p.add_argument("--num-blocks", type=int, default=14)
    p.add_argument("--calib-n", type=int, default=32)
    p.add_argument("--lambada-n", type=int, default=500)
    p.add_argument("--quantile", type=float, default=0.85)
    p.add_argument("--eligible", default=None,
                   help="comma-separated block ids allowed to skip; default = all skippable")
    p.add_argument("--max-skips", type=int, default=2)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    device = f"cuda:{args.gpu}"

    model, tok = load_trained(args.state_path, device, args.num_blocks)
    eligible = None
    if args.eligible:
        eligible = [int(x) for x in args.eligible.split(",")]
    else:
        eligible = list(model.skippable_blocks)

    print(f"[dynamic-skip] Calibrating w_recent on {args.calib_n} samples...")
    per_block = calibrate_thresholds(model, tok, device, n_samples=args.calib_n)
    thresholds = pick_thresholds(per_block, q=args.quantile)
    print(f"[dynamic-skip] Per-block τ at q={args.quantile}:")
    for b in sorted(thresholds.keys()):
        samples = per_block.get(b, [])
        if not samples:
            continue
        print(f"  block {b:2d}: τ={thresholds[b]:.4f}  "
              f"range=[{min(samples):.3f},{max(samples):.3f}]  n={len(samples)}")

    print(f"\n[dynamic-skip] Eval LAMBADA (n={args.lambada_n}, "
          f"eligible={eligible}, max_skips={args.max_skips}, q={args.quantile})")
    acc, ppl, avg_sk = eval_dynamic_lambada(
        model, tok, device, eligible, thresholds, args.max_skips, n=args.lambada_n)
    print(f"\n=== DYNAMIC SKIP [q={args.quantile}, P={eligible}, M={args.max_skips}] ===")
    print(f"LAMBADA acc: {acc:.4f}  ppl: {ppl:.3f}  avg_skips_per_input: {avg_sk:.2f}/{len(eligible)}")


if __name__ == "__main__":
    main()
