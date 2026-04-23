"""MMMU validation eval for Qwen3VLAttnResRetrofit (base / init / trained).

Picks letter (A/B/C/D) with max length-normalized log-likelihood over options.
Skips multi-image questions (>1 image) for simplicity.
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

DEFAULT_SUBJECTS = [
    "Accounting", "Art", "Biology", "Chemistry", "Computer_Science",
    "Economics", "Geography", "History", "Math", "Physics",
]


def load_model(model_type, state_path, device, num_blocks=14):
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    if model_type == "base":
        base.eval()
        return base, processor, "base"
    kwargs = dict(num_blocks=num_blocks)
    if model_type == "init":
        model = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype)
        model.eval()
        return model, processor, "init(γ=0)"
    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {})
    if "adapter_rank" in cfg: kwargs["adapter_rank"] = cfg["adapter_rank"]
    if "num_blocks" in cfg: kwargs["num_blocks"] = cfg["num_blocks"]
    model = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype)
    model.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    model.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    model.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    model.eval()
    gmax = float(model.gamma.detach().abs().max())
    return model, processor, f"trained(γmax={gmax:.3f})"


@torch.no_grad()
def forward_logits(model, inputs, skip_blocks=None):
    if isinstance(model, Qwen3VLAttnResRetrofit):
        kw = {}
        if skip_blocks:
            kw["skip_block_indices"] = skip_blocks
        return model(**inputs, **kw).logits
    return model(**inputs, use_cache=False).logits


@torch.no_grad()
def score_option(model, processor, device, messages, images, opt_text, skip_blocks):
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = prefix + opt_text
    full_inputs = processor(text=[full], images=images or None, return_tensors="pt")
    prefix_inputs = processor(text=[prefix], images=images or None, return_tensors="pt")
    prefix_len = prefix_inputs["input_ids"].shape[1]
    full_len = full_inputs["input_ids"].shape[1]
    if full_len <= prefix_len:
        return -1e9
    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
    logits = forward_logits(model, full_inputs, skip_blocks)
    start = prefix_len - 1
    tgt = full_inputs["input_ids"][0, prefix_len:]
    pred = logits[0, start:start + len(tgt), :]
    lp = F.log_softmax(pred.float(), dim=-1)
    ll = lp.gather(1, tgt.unsqueeze(-1)).sum().item()
    return ll / len(tgt)


@torch.no_grad()
def eval_mmmu(model, processor, device, subjects, n_per_subject, skip_blocks):
    correct = total = skipped = 0
    t0 = time.time()
    for subj in subjects:
        try:
            ds = load_dataset("MMMU/MMMU", subj, split="validation")
        except Exception as e:
            print(f"  skip {subj}: {e}", flush=True); continue
        for i, ex in enumerate(ds):
            if i >= n_per_subject: break
            opts = ex["options"]
            if isinstance(opts, str):
                import ast
                try: opts = ast.literal_eval(opts)
                except Exception: skipped += 1; continue
            if not isinstance(opts, list) or len(opts) < 2:
                skipped += 1; continue
            imgs = [ex.get(f"image_{k}") for k in range(1, 8)]
            imgs = [x for x in imgs if x is not None]
            if len(imgs) != 1:
                skipped += 1; continue
            q = ex["question"]
            user_content = [
                {"type": "image", "image": imgs[0]},
                {"type": "text", "text": f"{q}\n\nAnswer with the correct letter (A/B/C/D)."},
            ]
            messages = [{"role": "user", "content": user_content}]
            scores = []
            for j, opt in enumerate(opts):
                letter = chr(ord("A") + j)
                try:
                    scores.append(score_option(
                        model, processor, device, messages, imgs,
                        f" {letter}. {opt}", skip_blocks))
                except Exception:
                    scores.append(-1e9)
            pred = chr(ord("A") + max(range(len(scores)), key=lambda k: scores[k]))
            gold = str(ex["answer"]).strip()
            correct += int(pred == gold); total += 1
        print(f"  {subj}: acc={correct/max(total,1):.4f} "
              f"(n={total}, skip={skipped}, {time.time()-t0:.0f}s)", flush=True)
    return correct, total, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["base", "init", "trained"], required=True)
    p.add_argument("--state-path", default=None)
    p.add_argument("--num-blocks", type=int, default=14)
    p.add_argument("--n-per-subject", type=int, default=10)
    p.add_argument("--subjects", default=None)
    p.add_argument("--skip-blocks", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--label", default=None)
    args = p.parse_args()
    device = f"cuda:{args.gpu}"
    subjects = args.subjects.split(",") if args.subjects else DEFAULT_SUBJECTS
    skip_blocks = [int(x) for x in args.skip_blocks.split(",")] if args.skip_blocks else None
    model, processor, label = load_model(args.model_type, args.state_path, device, args.num_blocks)
    if args.label: label = args.label
    if skip_blocks: label = f"{label}+skip{skip_blocks}"
    print(f"\n{'='*70}\nMMMU: {label}  (subjects={len(subjects)}, n/subj={args.n_per_subject})\n{'='*70}", flush=True)
    c, t, sk = eval_mmmu(model, processor, device, subjects, args.n_per_subject, skip_blocks)
    print(f"\n=== MMMU [{label}] ===", flush=True)
    print(f"accuracy: {c}/{t} = {c/max(t,1):.4f}  (skipped {sk})", flush=True)


if __name__ == "__main__":
    main()
