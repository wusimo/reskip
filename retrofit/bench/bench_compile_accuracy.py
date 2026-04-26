"""Verify torch.compile preserves retrofit logits.

For each input, runs eager + compiled forward and reports
- argmax agreement (% tokens where eager and compiled pick same top-1)
- max abs logit delta over the prompt
- top-1 logit MSE

If compile is correct, argmax agreement should be 100% and logit deltas
should be tiny (bf16 numeric noise, ~0.1-0.4 max).
"""
from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoTokenizer
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit

MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"
PROMPTS = [
    "The capital of France is",
    "Einstein's most famous equation states that",
    "In Python, a list comprehension is written as",
    "The first president of the United States was",
    "A polynomial of degree two has at most",
    "Deep learning differs from classical machine learning in that",
    "When water boils at sea level, its temperature is approximately",
    "Recursion in computer science means that a function",
]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--state-path", required=True)
    ap.add_argument("--compile-mode", default="reduce-overhead")
    ap.add_argument("--seq-len", type=int, default=64,
                    help="pad inputs to this length so compiled (fixed-shape) graph hits one buckets")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[parity] state: {args.state_path}", flush=True)
    print(f"[parity] compile mode: {args.compile_mode}, seq_len={args.seq_len}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    ck = torch.load(args.state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kw = dict(num_blocks=cfg.get("num_blocks", 14))
    if "adapter_rank" in cfg:
        kw["adapter_rank"] = cfg["adapter_rank"]
    retro = Qwen3VLAttnResRetrofit(base, **kw).to(device=device, dtype=dtype).eval()
    retro.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    retro.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    retro.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))

    print("[parity] compiling …", flush=True)
    retro_compiled = torch.compile(retro, mode=args.compile_mode, dynamic=False)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    total_real = 0
    argmax_agree_real = 0
    total_pad = 0
    argmax_agree_pad = 0
    max_delta_real = 0.0
    sse_real = 0.0
    sse_real_n = 0

    for i, prompt in enumerate(PROMPTS):
        ids_real = tok.encode(prompt, add_special_tokens=False)
        L = len(ids_real)
        if L < args.seq_len:
            ids = ids_real + [pad_id] * (args.seq_len - L)
        else:
            ids = ids_real[: args.seq_len]
            L = args.seq_len
        inp = torch.tensor([ids], device=device)
        # Call the wrapped forward (retro / retro_compiled), NOT
        # ``.base_model``. ``retro_compiled.base_model`` is attribute
        # access and dispatches to the raw eager base — would hide any
        # compile divergence.
        out_eager = retro(input_ids=inp, use_cache=True)
        out_comp = retro_compiled(input_ids=inp, use_cache=True)

        le = out_eager.logits[0]   # [T, V]
        lc = out_comp.logits[0]
        ae = le.argmax(-1)
        ac = lc.argmax(-1)
        # Real prompt prediction tokens are positions [0, L-1] (each predicts
        # the next token; the very last position predicts the first pad).
        # Compare argmax there. Pad-region tokens (positions L..seq_len) are
        # garbage prediction targets and reflect mostly compile numeric noise,
        # so we report them separately as a noise floor.
        agree_real = (ae[:L] == ac[:L]).sum().item()
        agree_pad = (ae[L:] == ac[L:]).sum().item()
        delta_real = (le[:L].float() - lc[:L].float()).abs().max().item()
        sse_real += ((le[:L].float() - lc[:L].float()) ** 2).sum().item()
        sse_real_n += le[:L].numel()
        total_real += L
        argmax_agree_real += agree_real
        total_pad += le.shape[0] - L
        argmax_agree_pad += agree_pad
        max_delta_real = max(max_delta_real, delta_real)
        print(f"  [{i+1}/{len(PROMPTS)}] real={L} agree_real={agree_real}/{L} "
              f"agree_pad={agree_pad}/{le.shape[0]-L} max_delta_real={delta_real:.4f}",
              flush=True)

    rmse_real = (sse_real / max(sse_real_n, 1)) ** 0.5
    print(f"\n=== Compile parity (state={args.state_path.split('/')[-2]}, mode={args.compile_mode}) ===")
    print(f"  REAL prompt tokens (where the answer matters):")
    print(f"    argmax agreement: {argmax_agree_real}/{total_real} = "
          f"{argmax_agree_real/max(total_real,1)*100:.2f}%")
    print(f"    max |logit_eager - logit_compiled|: {max_delta_real:.4f}")
    print(f"    logit RMSE: {rmse_real:.6f}")
    print(f"  PAD region tokens (noise floor / numerics-sensitivity check):")
    print(f"    argmax agreement: {argmax_agree_pad}/{total_pad} = "
          f"{argmax_agree_pad/max(total_pad,1)*100:.2f}%")
    if argmax_agree_real == total_real:
        print("  PASS: argmax bit-equal on REAL prompt tokens")
    else:
        print(f"  WARN: {total_real - argmax_agree_real} REAL tokens differ "
              f"between eager and compiled")


if __name__ == "__main__":
    main()
