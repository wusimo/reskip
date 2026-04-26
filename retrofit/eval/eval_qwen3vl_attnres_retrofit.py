"""Eval for Qwen3VLAttnResRetrofit: LAMBADA + HellaSwag (+ optional MMMU).

Supports three model modes:
  - base:      original Qwen3-VL-2B
  - init:      Qwen3VLAttnResRetrofit wrapper at init (γ=0, random router) —
               must match base (identity-at-init test via downstream metrics).
  - trained:   load retrofit_attnres_state.pt into the wrapper. Also supports
               skip evaluation via --skip-blocks.
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
from transformers import AutoModelForImageTextToText, AutoTokenizer
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit
from compile_utils import wrap_compile, add_compile_arg


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load_model(model_type, state_path, device, num_blocks=14,
               skippable_blocks=None, adapter_rank=128, model_path=None):
    dtype = torch.bfloat16
    mp = model_path or MODEL_PATH
    tok = AutoTokenizer.from_pretrained(mp)
    base = AutoModelForImageTextToText.from_pretrained(mp, dtype=dtype).to(device)

    if model_type == "base":
        base.eval()
        return base, tok, "base", None

    kwargs = dict(num_blocks=num_blocks, adapter_rank=adapter_rank)
    if skippable_blocks is not None:
        kwargs["skippable_blocks"] = skippable_blocks

    if model_type == "init":
        model = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype)
        model.eval()
        return model, tok, "init(γ=0)", model

    # trained
    ck = torch.load(state_path, map_location="cpu")
    skippable = ck.get("skippable_blocks")
    cfg = ck.get("config", {})
    if skippable is not None:
        kwargs["skippable_blocks"] = skippable
    if "adapter_rank" in cfg:
        kwargs["adapter_rank"] = cfg["adapter_rank"]
    if "num_blocks" in cfg:
        kwargs["num_blocks"] = cfg["num_blocks"]
    # detect no-adapter training (Identity adapters have empty state_dict)
    no_adapter = cfg.get("no_adapter", not ck["adapters"])
    if no_adapter:
        kwargs["no_adapter"] = True
        kwargs.pop("adapter_rank", None)
    model = Qwen3VLAttnResRetrofit(base, **kwargs).to(device=device, dtype=dtype)
    model.router.load_state_dict(
        {k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    if not no_adapter:
        model.adapters.load_state_dict(
            {k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    model.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    model.eval()
    gmax = float(model.gamma.detach().abs().max())
    gmean = float(model.gamma.detach().mean())
    return model, tok, f"trained(γ_max={gmax:.3f},mean={gmean:+.3f})", model


@torch.no_grad()
def forward_logits(model, ids, skip_block_indices=None, is_retrofit=False):
    """is_retrofit: True when ``model`` wraps a Qwen3VLAttnResRetrofit
    (also after torch.compile, which would otherwise break an isinstance check)."""
    if is_retrofit:
        kwargs = {}
        if skip_block_indices:
            kwargs["skip_block_indices"] = skip_block_indices
        out = model(input_ids=ids, **kwargs)
        return out.logits
    return model(input_ids=ids, use_cache=False).logits


@torch.no_grad()
def eval_lambada(model, tok, device, n=500, skip_block_indices=None, is_retrofit=False):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if n < len(ds):
        ds = ds.select(range(n))
    correct = 0; total = 0; nll = 0.0; ntok = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        text = ex["text"].strip()
        if " " not in text:
            continue
        last = text.rfind(" ")
        ctx = text[:last]
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        full_ids = tok.encode(text, add_special_tokens=False)
        tgt_ids = full_ids[len(ctx_ids):]
        if not tgt_ids:
            continue
        inp = torch.tensor([ctx_ids + tgt_ids], device=device)
        logits = forward_logits(model, inp, skip_block_indices, is_retrofit=is_retrofit)
        start = len(ctx_ids) - 1
        pred_logits = logits[0, start:start + len(tgt_ids), :]
        tgt = torch.tensor(tgt_ids, device=device)
        is_corr = bool((pred_logits.argmax(-1) == tgt).all().item())
        lp = F.log_softmax(pred_logits.float(), dim=-1)
        nll += -lp.gather(1, tgt.unsqueeze(-1)).sum().item()
        ntok += len(tgt_ids)
        correct += int(is_corr); total += 1
        if (i + 1) % 100 == 0:
            print(f"  lambada {i+1}/{len(ds)} acc={correct/total:.4f} "
                  f"ppl={math.exp(nll/ntok):.2f} ({time.time()-t0:.0f}s)", flush=True)
    return correct / max(total, 1), math.exp(nll / max(ntok, 1)), total


@torch.no_grad()
def eval_hellaswag(model, tok, device, n=500, skip_block_indices=None, is_retrofit=False):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if n < len(ds):
        ds = ds.select(range(n))
    correct = 0; total = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        ctx = ex["ctx"]; endings = ex["endings"]; label = int(ex["label"])
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        scores = []
        for end in endings:
            full_ids = tok.encode(ctx + " " + end, add_special_tokens=False)
            tgt_ids = full_ids[len(ctx_ids):]
            if not tgt_ids:
                scores.append(-1e9); continue
            inp = torch.tensor([ctx_ids + tgt_ids], device=device)
            logits = forward_logits(model, inp, skip_block_indices, is_retrofit=is_retrofit)
            start = len(ctx_ids) - 1
            pred_logits = logits[0, start:start + len(tgt_ids), :]
            lp = F.log_softmax(pred_logits.float(), dim=-1)
            tgt = torch.tensor(tgt_ids, device=device)
            ll = lp.gather(1, tgt.unsqueeze(-1)).sum().item()
            scores.append(ll / len(tgt_ids))
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        correct += int(pred == label); total += 1
        if (i + 1) % 100 == 0:
            print(f"  hellaswag {i+1}/{len(ds)} acc_norm={correct/total:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return correct / max(total, 1), total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["base", "init", "trained"], required=True)
    p.add_argument("--state-path", default=None)
    p.add_argument("--num-blocks", type=int, default=14)
    p.add_argument("--adapter-rank", type=int, default=128)
    p.add_argument("--lambada-n", type=int, default=500)
    p.add_argument("--hellaswag-n", type=int, default=500)
    p.add_argument("--skip-hellaswag", action="store_true")
    p.add_argument("--skip-blocks", default=None,
                   help="comma-separated block ids to skip at inference (trained only)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--label", default=None)
    p.add_argument("--model-path", default=None,
                   help="HF model dir. Default: Qwen3-VL-2B (MODEL_PATH constant).")
    add_compile_arg(p)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    model, tok, label, retrofit = load_model(
        args.model_type, args.state_path, device,
        num_blocks=args.num_blocks, adapter_rank=args.adapter_rank,
        model_path=args.model_path,
    )
    if args.label:
        label = args.label
    skip_blocks = [int(x) for x in args.skip_blocks.split(",")] if args.skip_blocks else None
    if skip_blocks:
        label = f"{label}+skip{skip_blocks}"
    # Wrap with torch.compile per the iso-cost paper claim. Variable-length
    # LAMBADA / HellaSwag inputs → dynamic=True so the compile cache doesn't
    # thrash. Skip-blocks path uses the retrofit wrapper which has a runtime
    # branch; compile still works (graph-breaks at the branch but the bulk
    # of layer kernels remain compiled). Use --compile-mode off to bypass.
    is_retrofit = retrofit is not None
    model = wrap_compile(model, mode=args.compile_mode, label=f"{label}")
    print(f"\n{'='*70}\n{label}\n{'='*70}", flush=True)

    acc, ppl, tot = eval_lambada(model, tok, device, n=args.lambada_n,
                                 skip_block_indices=skip_blocks,
                                 is_retrofit=is_retrofit)
    print(f"LAMBADA: acc={acc:.4f} ppl={ppl:.3f} n={tot}", flush=True)

    hs_acc = None
    if not args.skip_hellaswag:
        hs_acc, hs_tot = eval_hellaswag(model, tok, device, n=args.hellaswag_n,
                                        skip_block_indices=skip_blocks,
                                        is_retrofit=is_retrofit)
        print(f"HellaSwag: acc_norm={hs_acc:.4f} n={hs_tot}", flush=True)

    print(f"\n=== SUMMARY [{label}] ===", flush=True)
    print(f"LAMBADA acc: {acc:.4f}  ppl: {ppl:.3f}", flush=True)
    if hs_acc is not None:
        print(f"HellaSwag acc_norm: {hs_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
