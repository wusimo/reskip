"""MMStar eval for Qwen3VLAttnResRetrofit.

MMStar (Lin-Chen/MMStar) — 1.5k vision-essential MCQ, deduplicated against
training data. Modern replacement for MMBench for paper-quality eval.
Single image + A/B/C/D MCQ.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load_model(model_type, state_path, device, num_blocks=14, model_path=None):
    dtype = torch.bfloat16
    mp = model_path or MODEL_PATH
    processor = AutoProcessor.from_pretrained(mp)
    base = AutoModelForImageTextToText.from_pretrained(mp, dtype=dtype).to(device)
    if model_type == "base":
        base.eval()
        return base, processor, "base"
    kwargs = dict(num_blocks=num_blocks)
    if model_type == "init":
        m = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype)
        m.eval()
        return m, processor, "init(γ=0)"
    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {})
    if "adapter_rank" in cfg: kwargs["adapter_rank"] = cfg["adapter_rank"]
    if "num_blocks" in cfg: kwargs["num_blocks"] = cfg["num_blocks"]
    m = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype)
    m.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    m.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    m.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    m.eval()
    return m, processor, "trained"


@torch.no_grad()
def calibrate_dyn(model, tok, device, n=32, q=0.85):
    """Run LAMBADA prefixes, collect per-block w_recent, return τ_n at quantile q."""
    from collections import defaultdict
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    ds = ds.select(range(n, n + 32))
    per_block = defaultdict(list)
    for ex in ds:
        ids = tok.encode(ex["text"].strip(), add_special_tokens=False)[:512]
        inp = torch.tensor([ids], device=device)
        out = model(input_ids=inp, return_alpha=True)
        for i, tr in enumerate(out.skip_trace or []):
            w = tr.get("w_recent")
            if w is not None:
                per_block[i].append(w)
    thr = {}
    for b, vals in per_block.items():
        vals = sorted(vals)
        thr[b] = vals[int(q * (len(vals) - 1))]
    return thr


@torch.no_grad()
def forward_logits(model, inputs, skip_blocks=None, dynamic_cfg=None):
    if isinstance(model, Qwen3VLAttnResRetrofit):
        kw = {}
        if skip_blocks: kw["skip_block_indices"] = skip_blocks
        if dynamic_cfg: kw["dynamic_skip_config"] = dynamic_cfg
        return model(**inputs, **kw).logits
    return model(**inputs, use_cache=False).logits


@torch.no_grad()
def score_option(model, processor, device, messages, images, opt_text, skip_blocks, dynamic_cfg=None):
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = prefix + opt_text
    full_inputs = processor(text=[full], images=images or None, return_tensors="pt")
    prefix_inputs = processor(text=[prefix], images=images or None, return_tensors="pt")
    prefix_len = prefix_inputs["input_ids"].shape[1]
    full_len = full_inputs["input_ids"].shape[1]
    if full_len <= prefix_len:
        return -1e9
    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
    logits = forward_logits(model, full_inputs, skip_blocks, dynamic_cfg)
    start = prefix_len - 1
    tgt = full_inputs["input_ids"][0, prefix_len:]
    pred = logits[0, start:start + len(tgt), :]
    lp = F.log_softmax(pred.float(), dim=-1)
    ll = lp.gather(1, tgt.unsqueeze(-1)).sum().item()
    return ll / len(tgt)


@torch.no_grad()
def eval_mmstar(model, processor, device, n, skip_blocks, dynamic_cfg=None):
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    if n < len(ds):
        ds = ds.select(range(n))
    correct = total = skipped = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        img = ex.get("image")
        if img is None:
            skipped += 1; continue
        q = ex.get("question", "")
        # MMStar question contains options A. x  B. y  ... inline
        gold = str(ex.get("answer", "")).strip().upper()
        if gold not in "ABCDEF":
            skipped += 1; continue
        # Extract options A,B,C,D from question text
        import re
        opts = []
        for L in "ABCDEF":
            m = re.search(rf"(?:^|\n|\s){L}[.\):\-]\s*([^\n]*?)(?=(?:\n[A-F][.\):\-])|$)", q)
            if m:
                opts.append((L, m.group(1).strip()))
        if len(opts) < 2:
            # Fallback — just score letters on question as-is
            user = [
                {"type": "image", "image": img},
                {"type": "text", "text": f"{q}\n\nAnswer with the correct letter (A/B/C/D)."},
            ]
            messages = [{"role": "user", "content": user}]
            scores = []
            for L in "ABCD":
                try:
                    scores.append(score_option(model, processor, device, messages, [img],
                                               f" {L}", skip_blocks, dynamic_cfg))
                except Exception:
                    scores.append(-1e9)
            pred = chr(ord("A") + max(range(len(scores)), key=lambda k: scores[k]))
        else:
            user = [
                {"type": "image", "image": img},
                {"type": "text", "text": f"{q}\n\nAnswer with the correct letter."},
            ]
            messages = [{"role": "user", "content": user}]
            scores = []
            for L, v in opts:
                try:
                    scores.append(score_option(model, processor, device, messages, [img],
                                               f" {L}. {v}", skip_blocks, dynamic_cfg))
                except Exception:
                    scores.append(-1e9)
            pred = opts[max(range(len(scores)), key=lambda k: scores[k])][0]
        correct += int(pred == gold); total += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)} acc={correct/total:.4f} skip={skipped} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return correct, total, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["base", "init", "trained"], required=True)
    p.add_argument("--state-path", default=None)
    p.add_argument("--num-blocks", type=int, default=14)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--skip-blocks", default=None)
    p.add_argument("--dyn-quantile", type=float, default=None,
                   help="enable dynamic skip with this quantile (e.g., 0.85)")
    p.add_argument("--dyn-max-skips", type=int, default=2)
    p.add_argument("--dyn-eligible", default=None,
                   help="comma-separated block ids for dynamic eligible set")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--label", default=None)
    p.add_argument("--model-path", default=None, help="HF model dir; default Qwen3-VL-2B")
    args = p.parse_args()
    device = f"cuda:{args.gpu}"
    model, processor, label = load_model(args.model_type, args.state_path, device, args.num_blocks, model_path=args.model_path)
    if args.label: label = args.label
    skip_blocks = [int(x) for x in args.skip_blocks.split(",")] if args.skip_blocks else None
    if skip_blocks: label = f"{label}+skip{skip_blocks}"
    dynamic_cfg = None
    if args.dyn_quantile is not None and isinstance(model, Qwen3VLAttnResRetrofit):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path or MODEL_PATH)
        print(f"[mmstar] Calibrating dynamic skip q={args.dyn_quantile}...", flush=True)
        thr = calibrate_dyn(model, tok, device, q=args.dyn_quantile)
        eligible = None
        if args.dyn_eligible:
            eligible = set(int(x) for x in args.dyn_eligible.split(","))
        else:
            eligible = set(model.skippable_blocks)
        dynamic_cfg = dict(thresholds=thr, eligible_blocks=eligible, max_skips=args.dyn_max_skips)
        label = f"{label}+dyn(q={args.dyn_quantile},M={args.dyn_max_skips},P={sorted(eligible)})"
    print(f"\n{'='*70}\nMMStar: {label}  (n={args.n})\n{'='*70}", flush=True)
    c, t, sk = eval_mmstar(model, processor, device, args.n, skip_blocks, dynamic_cfg)
    print(f"\n=== MMStar [{label}] ===", flush=True)
    print(f"accuracy: {c}/{t} = {c/max(t,1):.4f}  (skipped {sk})", flush=True)


if __name__ == "__main__":
    main()
