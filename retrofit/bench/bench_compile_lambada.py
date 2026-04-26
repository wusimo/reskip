"""Run LAMBADA on retrofit under torch.compile to validate the
accuracy-preservation claim downstream of the per-token logit parity
test (`bench_compile_accuracy.py`).

LAMBADA examples have variable lengths so we use `mode="default",
dynamic=True` (no CUDA-graph capture, but inductor still traces and
fuses kernels). Compares retrofit eager vs retrofit compiled at the
same N=500 examples; reports both accuracies and the per-example
argmax agreement.
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

MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load_retro(state_path, device, dtype):
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kw = dict(num_blocks=cfg.get("num_blocks", 14))
    if "adapter_rank" in cfg:
        kw["adapter_rank"] = cfg["adapter_rank"]
    retro = Qwen3VLAttnResRetrofit(base, **kw).to(device=device, dtype=dtype).eval()
    retro.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    retro.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    retro.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    return retro


@torch.no_grad()
def eval_lambada_pair(retro_eager, retro_compiled, tok, device, n=500):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if n < len(ds):
        ds = ds.select(range(n))
    ce, cc, total = 0, 0, 0
    last_tok_match = 0
    nll_e, nll_c, ntok = 0.0, 0.0, 0
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
        out_e = retro_eager(input_ids=inp, use_cache=False)
        out_c = retro_compiled(input_ids=inp, use_cache=False)
        le = out_e.logits[0]
        lc = out_c.logits[0]
        start = len(ctx_ids) - 1
        pe = le[start:start + len(tgt_ids)]
        pc = lc[start:start + len(tgt_ids)]
        tgt = torch.tensor(tgt_ids, device=device)
        ie = bool((pe.argmax(-1) == tgt).all().item())
        ic = bool((pc.argmax(-1) == tgt).all().item())
        ce += int(ie); cc += int(ic); total += 1
        last_tok_match += int(pe.argmax(-1).tolist() == pc.argmax(-1).tolist())
        nll_e += -F.log_softmax(pe.float(), dim=-1).gather(1, tgt.unsqueeze(-1)).sum().item()
        nll_c += -F.log_softmax(pc.float(), dim=-1).gather(1, tgt.unsqueeze(-1)).sum().item()
        ntok += len(tgt_ids)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)} eager={ce/total:.4f} compiled={cc/total:.4f} "
                  f"argmax-agree={last_tok_match/total:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
    return {
        "n": total,
        "acc_eager": ce / max(total, 1),
        "acc_compiled": cc / max(total, 1),
        "ppl_eager": math.exp(nll_e / max(ntok, 1)),
        "ppl_compiled": math.exp(nll_c / max(ntok, 1)),
        "target_argmax_agreement": last_tok_match / max(total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--state-path", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--compile-mode", default="default",
                    help="default = traces/fuses without CUDA-graph; reduce-overhead "
                         "needs fixed shapes (won't work cleanly with variable LAMBADA inputs).")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[lambada] state: {args.state_path}", flush=True)
    print(f"[lambada] n={args.n}, compile-mode={args.compile_mode}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    retro = load_retro(args.state_path, device, dtype)
    retro_compiled = torch.compile(retro, mode=args.compile_mode, dynamic=True)
    res = eval_lambada_pair(retro, retro_compiled, tok, device, n=args.n)
    print("\n=== LAMBADA-{n} compile-vs-eager (state={ckpt}, mode={mode}) ===".format(
        n=res["n"], ckpt=args.state_path.split("/")[-2], mode=args.compile_mode))
    print(f"  acc eager:    {res['acc_eager']:.4f}  ppl eager:    {res['ppl_eager']:.3f}")
    print(f"  acc compiled: {res['acc_compiled']:.4f}  ppl compiled: {res['ppl_compiled']:.3f}")
    print(f"  Δacc = {(res['acc_compiled'] - res['acc_eager'])*100:+.2f} pp")
    print(f"  per-example target-argmax agreement: {res['target_argmax_agreement']*100:.2f}%")


if __name__ == "__main__":
    main()
