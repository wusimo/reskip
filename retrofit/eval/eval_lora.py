"""Eval for LoRA baseline: load base Qwen3-VL-2B + LoRA adapter, run LAMBADA
+ HellaSwag (matches eval_qwen3vl_attnres_retrofit.py format for direct
comparison)."""
from __future__ import annotations
import argparse
import math
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load_lora(adapter_path, device):
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, tok, f"LoRA({n_train/1e6:.1f}M)"


@torch.no_grad()
def eval_lambada(model, tok, device, n=500):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if n < len(ds): ds = ds.select(range(n))
    correct = total = 0; nll = 0.0; ntok = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        text = ex["text"].strip()
        if " " not in text: continue
        last = text.rfind(" ")
        ctx_ids = tok.encode(text[:last], add_special_tokens=False)
        full_ids = tok.encode(text, add_special_tokens=False)
        tgt_ids = full_ids[len(ctx_ids):]
        if not tgt_ids: continue
        inp = torch.tensor([ctx_ids + tgt_ids], device=device)
        logits = model(input_ids=inp, use_cache=False).logits
        start = len(ctx_ids) - 1
        pred = logits[0, start:start + len(tgt_ids), :]
        tgt = torch.tensor(tgt_ids, device=device)
        is_corr = bool((pred.argmax(-1) == tgt).all().item())
        lp = F.log_softmax(pred.float(), dim=-1)
        nll += -lp.gather(1, tgt.unsqueeze(-1)).sum().item()
        ntok += len(tgt_ids)
        correct += int(is_corr); total += 1
        if (i + 1) % 100 == 0:
            print(f"  lambada {i+1}/{len(ds)} acc={correct/total:.4f} "
                  f"ppl={math.exp(nll/ntok):.2f} ({time.time()-t0:.0f}s)", flush=True)
    return correct/max(total,1), math.exp(nll/max(ntok,1)), total


@torch.no_grad()
def eval_hellaswag(model, tok, device, n=500):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if n < len(ds): ds = ds.select(range(n))
    correct = total = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        ctx = ex["ctx"]; ends = ex["endings"]; label = int(ex["label"])
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        scores = []
        for end in ends:
            full_ids = tok.encode(ctx + " " + end, add_special_tokens=False)
            tgt_ids = full_ids[len(ctx_ids):]
            if not tgt_ids: scores.append(-1e9); continue
            inp = torch.tensor([ctx_ids + tgt_ids], device=device)
            logits = model(input_ids=inp, use_cache=False).logits
            start = len(ctx_ids) - 1
            pred = logits[0, start:start + len(tgt_ids), :]
            lp = F.log_softmax(pred.float(), dim=-1)
            tgt = torch.tensor(tgt_ids, device=device)
            ll = lp.gather(1, tgt.unsqueeze(-1)).sum().item()
            scores.append(ll / len(tgt_ids))
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        correct += int(pred == label); total += 1
        if (i + 1) % 100 == 0:
            print(f"  hellaswag {i+1}/{len(ds)} acc={correct/total:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return correct/max(total,1), total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--lambada-n", type=int, default=500)
    p.add_argument("--hellaswag-n", type=int, default=500)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--label", default=None)
    args = p.parse_args()
    device = f"cuda:{args.gpu}"
    model, tok, label = load_lora(args.adapter_path, device)
    if args.label: label = args.label
    print(f"\n{'='*70}\n{label}\n{'='*70}", flush=True)
    acc, ppl, tot = eval_lambada(model, tok, device, n=args.lambada_n)
    print(f"LAMBADA acc: {acc:.4f} ppl: {ppl:.3f} n={tot}", flush=True)
    hs, hs_tot = eval_hellaswag(model, tok, device, n=args.hellaswag_n)
    print(f"HellaSwag acc_norm: {hs:.4f} n={hs_tot}", flush=True)
    print(f"\n=== SUMMARY [{label}] ===")
    print(f"LAMBADA acc: {acc:.4f}  ppl: {ppl:.3f}")
    print(f"HellaSwag: {hs:.4f}")


if __name__ == "__main__":
    main()
