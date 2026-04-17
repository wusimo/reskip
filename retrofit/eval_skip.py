"""Evaluate retrofit model's skip Pareto.

Two things tested:
  1. β-sweep: vary Route A's gate β [0.1, 0.3, 0.5, 0.7, 1.0] → ppl
     (higher β = more reliance on AttnRes, closer to "pure AttnRes" inference)
  2. Block-skip: identify low-importance blocks from α analysis and test
     inference with those blocks skipped (standard ReSkip style)
"""
from __future__ import annotations
import argparse
import math
import sys
from copy import deepcopy

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flash-linear-attention")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flame")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/experiments")

import fla  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer
from flame_reskip_common import build_text_dataloader, count_valid_tokens
from retrofit_model import RetrofitModel


MODEL_PATH = "/home/user01/Minko/reskip2/reskip/flame/saves/transformer_test"
DATA_PATH = "/home/user01/Minko/datasets/fineweb_edu_100BT"


def load_retrofit(checkpoint_path, route, num_blocks=6, device="cuda:0", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=dtype
    ).to(device)
    model = RetrofitModel(base, num_blocks=num_blocks, route=route).to(device=device, dtype=dtype)
    if route == "A":
        model.gate_logits.data = model.gate_logits.data.to(dtype)
    ck = torch.load(checkpoint_path, map_location=device)
    model.router.load_state_dict(ck["router"])
    if route == "A" and "gate_logits" in ck:
        model.gate_logits.data.copy_(ck["gate_logits"].to(device=device, dtype=dtype))
    if "base_state_dict" in ck:
        model.base_model.load_state_dict(ck["base_state_dict"])
    model.eval()
    return model, tok


@torch.no_grad()
def measure_ppl(model, tok, device, num_batches=8, seq_len=8192):
    model.eval()
    dl = build_text_dataloader(
        tokenizer=tok, dataset=DATA_PATH, dataset_name=None, dataset_split="train",
        data_dir=None, data_files=None, seq_len=seq_len, context_len=2048,
        batch_size=1, num_workers=2, streaming=True, varlen=True, seed=0,
    )
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(dl):
        if i >= num_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        cu = batch.get("cu_seqlens")
        if cu is not None:
            cu = cu.to(device)
        out = model(input_ids=ids, labels=labels, cu_seqlens=cu)
        n = count_valid_tokens(labels)
        total_loss += float(out.loss) * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


def beta_sweep(model, tok, device, betas=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0), num_batches=8):
    """Vary beta in Route A and measure PPL."""
    assert model.route == "A"
    results = []
    for beta in betas:
        # override beta by setting logit = logit(beta)
        logit = math.log(beta / (1 - beta)) if beta < 1.0 else 10.0  # very positive → β→1
        with torch.no_grad():
            model.gate_logits.data.fill_(logit)
        ppl = measure_ppl(model, tok, device, num_batches=num_batches)
        print(f"  β={beta:.2f} → PPL={ppl:.3f}")
        results.append((beta, ppl))
    return results


@torch.no_grad()
def block_importance_from_alpha(model, tok, device, num_batches=4, seq_len=8192):
    """Estimate block importance I(n) = max downstream α for block n's output."""
    dl = build_text_dataloader(
        tokenizer=tok, dataset=DATA_PATH, dataset_name=None, dataset_split="train",
        data_dir=None, data_files=None, seq_len=seq_len, context_len=2048,
        batch_size=1, num_workers=2, streaming=True, varlen=True, seed=123,
    )
    num_blocks = model.num_blocks
    # matrix[src][dst] = α from source src at destination block dst
    matrix = torch.zeros(num_blocks + 1, num_blocks + 1)
    counts = torch.zeros(num_blocks + 1, num_blocks + 1)
    for i, batch in enumerate(dl):
        if i >= num_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        cu = batch.get("cu_seqlens")
        if cu is not None:
            cu = cu.to(device)
        out = model(input_ids=ids, labels=labels, cu_seqlens=cu, return_alpha=True)
        if out.alpha_list is None:
            continue
        # For Route A/C, alpha_list has entries for n=1..num_blocks-1
        # with each entry shape [B, T, n+1] (includes embedding + blocks 0..n-1)
        for idx, alpha in enumerate(out.alpha_list):
            dst = idx + 1  # position n+1 in my indexing (1-based dst for blocks 1..N-1)
            # src 0..num_src-1 map to: src 0 = embedding, src 1..src-1 = blocks 0..src-2
            # Actually: sources at dst are {embed, b0, b1, ..., b_{dst-1}}
            # so src_idx 0 = embed, src_idx k>=1 = block k-1
            num_src = alpha.shape[-1]  # = dst + 1... wait let me re-check
            # alpha is returned from self.router.route(position=n+1, completed_outputs)
            # completed has n+1 elements (embedding + blocks 0..n-1 where n is iteration index)
            # Actually for A at block n (iteration), we route with position=n+1, completed has n+1 elements too
            # num_src = n+1
            src_to_label = num_src  # number of sources at this position
            mean = alpha.float().mean(dim=(0, 1)).cpu()  # [num_src]
            dst_actual = src_to_label  # how many completed blocks were processed before this
            for src in range(num_src):
                matrix[src][dst_actual] += mean[src].item()
                counts[src][dst_actual] += 1
    counts_safe = counts.clamp_min(1.0)
    avg_matrix = matrix / counts_safe
    importance = []
    # For each block b (0..N-1), its source index is b+1
    for b in range(num_blocks):
        src_idx = b + 1
        # max downstream α
        downstream = avg_matrix[src_idx, src_idx + 1:]  # dst > src_idx
        imp = float(downstream.max().item()) if downstream.numel() > 0 else 0.0
        importance.append(imp)
    # Also embedding importance
    emb_importance = float(avg_matrix[0, 1:].max().item())
    return importance, emb_importance, avg_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--route", choices=["A", "B"], default="A")
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    model, tok = load_retrofit(args.checkpoint, args.route, num_blocks=args.num_blocks, device=device)

    # Baseline: no retrofit (load fresh base)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device).eval()

    def baseline_ppl():
        dl = build_text_dataloader(
            tokenizer=tok, dataset=DATA_PATH, dataset_name=None, dataset_split="train",
            data_dir=None, data_files=None, seq_len=8192, context_len=2048,
            batch_size=1, num_workers=2, streaming=True, varlen=True, seed=0,
        )
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for i, batch in enumerate(dl):
                if i >= args.num_batches:
                    break
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                cu = batch.get("cu_seqlens")
                if cu is not None:
                    cu = cu.to(device)
                out = base(input_ids=ids, labels=labels, cu_seqlens=cu, use_cache=False)
                n = count_valid_tokens(labels)
                total_loss += float(out.loss) * n
                total_tokens += n
        return math.exp(total_loss / total_tokens)

    base_ppl = baseline_ppl()
    print(f"Baseline (no retrofit) PPL: {base_ppl:.3f}")

    # Block importance
    importance, emb_imp, _matrix = block_importance_from_alpha(model, tok, device)
    print(f"\n{'=' * 60}")
    print(f"Block importance (learned α):")
    print(f"{'=' * 60}")
    print(f"  embedding: {emb_imp:.4f}")
    for i, imp in enumerate(importance):
        mark = " ← SKIPPABLE" if imp < 0.1 else ""
        print(f"  block {i}: {imp:.4f}{mark}")

    # β sweep (only for Route A)
    if args.route == "A":
        print(f"\n{'=' * 60}")
        print(f"β sweep (Route A inference behavior)")
        print(f"{'=' * 60}")
        results = beta_sweep(model, tok, device, num_batches=args.num_batches)
        print("\nSummary:")
        for beta, ppl in results:
            ratio = ppl / base_ppl
            print(f"  β={beta:.2f} PPL={ppl:.3f} ratio={ratio:.3f}x base")


if __name__ == "__main__":
    main()
