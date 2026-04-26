from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict

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
def forward_logits(
    model,
    ids: torch.Tensor,
    skip_block_indices: list[int] | None = None,
    dynamic_skip_config: dict | None = None,
):
    if isinstance(model, Qwen3VLAttnResRetrofit):
        kwargs = {}
        if skip_block_indices:
            kwargs["skip_block_indices"] = skip_block_indices
        if dynamic_skip_config is not None:
            kwargs["dynamic_skip_config"] = dynamic_skip_config
        out = model(input_ids=ids, **kwargs)
        return out.logits, out.skip_trace
    return model(input_ids=ids, use_cache=False).logits, None


@torch.no_grad()
def calibrate_w_recent(
    model: Qwen3VLAttnResRetrofit,
    tok,
    device: str,
    calib_n: int,
    seq_len: int,
    offset: int,
):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    start = min(offset, max(0, len(ds) - calib_n))
    ds = ds.select(range(start, min(start + calib_n, len(ds))))
    per_block = defaultdict(list)
    for ex in ds:
        ids = tok.encode(ex["text"].strip(), add_special_tokens=False)[:seq_len]
        if len(ids) < 8:
            continue
        inp = torch.tensor([ids], device=device)
        out = model(input_ids=inp, return_alpha=True)
        for trace in out.skip_trace or []:
            block_idx = trace["block_idx"]
            w_recent = trace.get("w_recent")
            if w_recent is not None:
                per_block[block_idx].append(float(w_recent))
    return dict(per_block)


def pick_thresholds(per_block_samples: dict[int, list[float]], quantile: float):
    thresholds = {}
    summary = {}
    for block_idx, vals in per_block_samples.items():
        if not vals:
            continue
        vals_sorted = sorted(vals)
        q_idx = min(len(vals_sorted) - 1, max(0, int(quantile * (len(vals_sorted) - 1))))
        threshold = float(vals_sorted[q_idx])
        thresholds[block_idx] = threshold
        summary[block_idx] = {
            "threshold": threshold,
            "min": float(vals_sorted[0]),
            "max": float(vals_sorted[-1]),
            "mean": float(sum(vals_sorted) / len(vals_sorted)),
            "n": len(vals_sorted),
        }
    return thresholds, summary


@torch.no_grad()
def eval_lambada(model, tok, device: str, n: int, dynamic_skip_config: dict | None = None):
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    ds = ds.select(range(min(n, len(ds))))
    correct = 0
    total = 0
    nll = 0.0
    ntok = 0
    skip_counts = []
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
        logits, skip_trace = forward_logits(model, inp, dynamic_skip_config=dynamic_skip_config)
        start = len(ctx_ids) - 1
        pred_logits = logits[0, start : start + len(tgt_ids), :]
        tgt = torch.tensor(tgt_ids, device=device)
        correct += int((pred_logits.argmax(-1) == tgt).all().item())
        lp = F.log_softmax(pred_logits.float(), dim=-1)
        nll += -lp.gather(1, tgt.unsqueeze(-1)).sum().item()
        ntok += len(tgt_ids)
        total += 1
        if skip_trace:
            skip_counts.append(sum(1 for item in skip_trace if item["skipped"]))
        if (i + 1) % 100 == 0:
            avg_skips = sum(skip_counts) / max(len(skip_counts), 1)
            print(
                f"lambada {i+1}/{len(ds)} acc={correct/max(total,1):.4f} ppl={math.exp(nll/max(ntok,1)):.3f} avg_skips={avg_skips:.2f} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return {
        "acc": correct / max(total, 1),
        "ppl": math.exp(nll / max(ntok, 1)),
        "n": total,
        "avg_skips": sum(skip_counts) / max(len(skip_counts), 1),
    }


@torch.no_grad()
def eval_hellaswag(model, tok, device: str, n: int, dynamic_skip_config: dict | None = None):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.select(range(min(n, len(ds))))
    correct = 0
    total = 0
    skip_counts = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        scores = []
        trace_for_choice = None
        for end in endings:
            full_ids = tok.encode(ctx + " " + end, add_special_tokens=False)
            tgt_ids = full_ids[len(ctx_ids):]
            if not tgt_ids:
                scores.append(-1e9)
                continue
            inp = torch.tensor([ctx_ids + tgt_ids], device=device)
            logits, skip_trace = forward_logits(model, inp, dynamic_skip_config=dynamic_skip_config)
            if trace_for_choice is None:
                trace_for_choice = skip_trace
            start = len(ctx_ids) - 1
            pred_logits = logits[0, start : start + len(tgt_ids), :]
            tgt = torch.tensor(tgt_ids, device=device)
            lp = F.log_softmax(pred_logits.float(), dim=-1)
            scores.append(lp.gather(1, tgt.unsqueeze(-1)).sum().item() / len(tgt_ids))
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        correct += int(pred == label)
        total += 1
        if trace_for_choice:
            skip_counts.append(sum(1 for item in trace_for_choice if item["skipped"]))
        if (i + 1) % 100 == 0:
            avg_skips = sum(skip_counts) / max(len(skip_counts), 1)
            print(
                f"hellaswag {i+1}/{len(ds)} acc_norm={correct/max(total,1):.4f} avg_skips={avg_skips:.2f} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    return {
        "acc_norm": correct / max(total, 1),
        "n": total,
        "avg_skips": sum(skip_counts) / max(len(skip_counts), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--eligible-blocks", default="11")
    parser.add_argument("--quantiles", default="0.85,0.9,0.95")
    parser.add_argument("--max-skips", type=int, default=1)
    parser.add_argument("--calib-n", type=int, default=32)
    parser.add_argument("--calib-seq-len", type=int, default=512)
    parser.add_argument("--calib-offset", type=int, default=100)
    parser.add_argument("--lambada-n", type=int, default=100)
    parser.add_argument("--hellaswag-n", type=int, default=100)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    eligible_blocks = [int(x) for x in args.eligible_blocks.split(",") if x.strip()]
    quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
    base, model, tok = load_trained_model(args.state_path, device)
    samples = calibrate_w_recent(model, tok, device, args.calib_n, args.calib_seq_len, args.calib_offset)

    results = {
        "state_path": args.state_path,
        "eligible_blocks": eligible_blocks,
        "base": {
            "lambada": eval_lambada(base, tok, device, args.lambada_n),
            "hellaswag": eval_hellaswag(base, tok, device, args.hellaswag_n),
        },
        "full": {
            "lambada": eval_lambada(model, tok, device, args.lambada_n),
            "hellaswag": eval_hellaswag(model, tok, device, args.hellaswag_n),
        },
        "dynamic": [],
    }

    for quantile in quantiles:
        thresholds, threshold_summary = pick_thresholds(samples, quantile)
        filtered_thresholds = {block_idx: thresholds[block_idx] for block_idx in eligible_blocks if block_idx in thresholds}
        dynamic_skip_config = {
            "thresholds": filtered_thresholds,
            "eligible_blocks": set(eligible_blocks),
            "max_skips": args.max_skips,
        }
        dyn_result = {
            "quantile": quantile,
            "threshold_summary": {str(k): threshold_summary[k] for k in filtered_thresholds},
            "lambada": eval_lambada(model, tok, device, args.lambada_n, dynamic_skip_config),
            "hellaswag": eval_hellaswag(model, tok, device, args.hellaswag_n, dynamic_skip_config),
        }
        results["dynamic"].append(dyn_result)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
