from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/rechange")
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load_trained_model(state_path: str, device: str):
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer

    reference = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    ckpt = torch.load(state_path, map_location="cpu")
    wrapped_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    model = Qwen3VLAttnResRetrofit(
        wrapped_base,
        num_blocks=ckpt.get("config", {}).get("num_blocks", 14),
        adapter_rank=ckpt.get("config", {}).get("adapter_rank", 128),
        skippable_blocks=ckpt.get("skippable_blocks"),
    ).to(device=device, dtype=dtype).eval()
    model.load_state_dict(ckpt["model"], strict=False)
    return reference, model, tokenizer


@torch.no_grad()
def forward_logits(model, ids: torch.Tensor, skip_block_indices: list[int] | None = None):
    if isinstance(model, Qwen3VLAttnResRetrofit):
        kwargs = {}
        if skip_block_indices:
            kwargs["skip_block_indices"] = skip_block_indices
        return model(input_ids=ids, **kwargs).logits
    return model(input_ids=ids, use_cache=False).logits


@torch.no_grad()
def eval_lambada(model, tok, device: str, n: int, skip_block_indices: list[int] | None = None):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    ds = ds.select(range(min(n, len(ds))))
    correct = 0
    total = 0
    nll = 0.0
    ntok = 0
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
        logits = forward_logits(model, inp, skip_block_indices)
        start = len(ctx_ids) - 1
        pred_logits = logits[0, start : start + len(tgt_ids), :]
        tgt = torch.tensor(tgt_ids, device=device)
        correct += int((pred_logits.argmax(-1) == tgt).all().item())
        lp = F.log_softmax(pred_logits.float(), dim=-1)
        nll += -lp.gather(1, tgt.unsqueeze(-1)).sum().item()
        ntok += len(tgt_ids)
        total += 1
        if (i + 1) % 100 == 0:
            print(
                f"lambada {i+1}/{len(ds)} acc={correct/max(total,1):.4f} ppl={math.exp(nll/max(ntok,1)):.3f} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return {"acc": correct / max(total, 1), "ppl": math.exp(nll / max(ntok, 1)), "n": total}


@torch.no_grad()
def eval_hellaswag(model, tok, device: str, n: int, skip_block_indices: list[int] | None = None):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.select(range(min(n, len(ds))))
    correct = 0
    total = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        scores = []
        for end in endings:
            full_ids = tok.encode(ctx + " " + end, add_special_tokens=False)
            tgt_ids = full_ids[len(ctx_ids):]
            if not tgt_ids:
                scores.append(-1e9)
                continue
            inp = torch.tensor([ctx_ids + tgt_ids], device=device)
            logits = forward_logits(model, inp, skip_block_indices)
            start = len(ctx_ids) - 1
            pred_logits = logits[0, start : start + len(tgt_ids), :]
            tgt = torch.tensor(tgt_ids, device=device)
            lp = F.log_softmax(pred_logits.float(), dim=-1)
            scores.append(lp.gather(1, tgt.unsqueeze(-1)).sum().item() / len(tgt_ids))
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        correct += int(pred == label)
        total += 1
        if (i + 1) % 100 == 0:
            print(
                f"hellaswag {i+1}/{len(ds)} acc_norm={correct/max(total,1):.4f} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return {"acc_norm": correct / max(total, 1), "n": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--skip-blocks", default="")
    parser.add_argument("--lambada-n", type=int, default=100)
    parser.add_argument("--hellaswag-n", type=int, default=100)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    skip_blocks = [int(x) for x in args.skip_blocks.split(",") if x.strip()]
    base, model, tok = load_trained_model(args.state_path, device)

    results = {
        "state_path": args.state_path,
        "base": {
            "lambada": eval_lambada(base, tok, device, args.lambada_n),
            "hellaswag": eval_hellaswag(base, tok, device, args.hellaswag_n),
        },
        "full": {
            "lambada": eval_lambada(model, tok, device, args.lambada_n),
            "hellaswag": eval_hellaswag(model, tok, device, args.hellaswag_n),
        },
    }
    if skip_blocks:
        results["skip"] = {
            "blocks": skip_blocks,
            "lambada": eval_lambada(model, tok, device, args.lambada_n, skip_blocks),
            "hellaswag": eval_hellaswag(model, tok, device, args.hellaswag_n, skip_blocks),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
