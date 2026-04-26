"""MMBench eval (dev split) for Qwen3VLAttnResRetrofit.

MMBench — comprehensive 20-dim VLM capability benchmark. Uses HF dataset
`lmms-lab/MMBench` or `OpenGVLab/MMBench`. Single-image A/B/C/D MCQ.
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


def load_model(model_type, state_path, device, num_blocks=14):
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
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
def forward_logits(model, inputs, skip_blocks=None, dynamic_cfg=None):
    if isinstance(model, Qwen3VLAttnResRetrofit):
        kw = {}
        if skip_blocks: kw["skip_block_indices"] = skip_blocks
        if dynamic_cfg: kw["dynamic_skip_config"] = dynamic_cfg
        return model(**inputs, **kw).logits
    return model(**inputs, use_cache=False).logits


@torch.no_grad()
def score_option(model, processor, device, messages, images, opt_text, skip_blocks, dynamic_cfg):
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
def eval_mmbench(model, processor, device, n, skip_blocks, dynamic_cfg):
    try:
        ds = load_dataset("lmms-lab/MMBench_EN", split="dev")
    except Exception:
        ds = load_dataset("lmms-lab/MMBench", "dev", split="dev")
    if n < len(ds):
        ds = ds.select(range(n))
    correct = total = skipped = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        img = ex.get("image")
        if img is None:
            skipped += 1; continue
        q = ex.get("question", "")
        opts = []
        letters = ["A", "B", "C", "D"]
        for letter in letters:
            v = ex.get(letter)
            if v and isinstance(v, str):
                opts.append((letter, v))
        if len(opts) < 2:
            skipped += 1; continue
        gold = str(ex.get("answer", "")).strip()
        hint = ex.get("hint") or ""
        prompt = f"{hint}\n{q}" if hint else q
        user = [
            {"type": "image", "image": img},
            {"type": "text", "text": f"{prompt}\n\n" + "\n".join(f"{L}. {v}" for L, v in opts)
                                     + "\n\nAnswer with the correct letter."},
        ]
        messages = [{"role": "user", "content": user}]
        scores = []
        for L, v in opts:
            try:
                scores.append(score_option(model, processor, device, messages, [img],
                                           f" {L}. {v}", skip_blocks, dynamic_cfg))
            except Exception:
                scores.append(-1e9)
        pred_letter = opts[max(range(len(scores)), key=lambda k: scores[k])][0]
        correct += int(pred_letter == gold); total += 1
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
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--label", default=None)
    args = p.parse_args()
    device = f"cuda:{args.gpu}"
    model, processor, label = load_model(args.model_type, args.state_path, device, args.num_blocks)
    if args.label: label = args.label
    skip_blocks = [int(x) for x in args.skip_blocks.split(",")] if args.skip_blocks else None
    if skip_blocks: label = f"{label}+skip{skip_blocks}"
    print(f"\n{'='*70}\nMMBench: {label}  (n={args.n})\n{'='*70}", flush=True)
    c, t, sk = eval_mmbench(model, processor, device, args.n, skip_blocks, None)
    print(f"\n=== MMBench [{label}] ===", flush=True)
    print(f"accuracy: {c}/{t} = {c/max(t,1):.4f}  (skipped {sk})", flush=True)


if __name__ == "__main__":
    main()
