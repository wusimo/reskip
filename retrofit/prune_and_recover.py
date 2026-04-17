"""α-guided pruning + recovery fine-tune.

Given a Route A retrofit with learned α signals:
  1. Identify low-importance blocks (I(n) < ε)
  2. Create a pruned variant where those blocks are replaced with identity
  3. Briefly fine-tune the pruned model to recover quality
  4. Measure PPL + FLOPs savings

The pruned model is the ORIGINAL transformer (no AttnRes) with specific
blocks turned into identity. Fine-tune trains remaining layers to adapt.
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flash-linear-attention")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flame")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/experiments")

import fla  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer
from flame_reskip_common import build_text_dataloader, count_valid_tokens


MODEL_PATH = "/home/user01/Minko/reskip2/reskip/flame/saves/transformer_test"
DATA_PATH = "/home/user01/Minko/datasets/fineweb_edu_100BT"


class PrunedTransformer(nn.Module):
    """Wraps the base transformer with specific layers replaced by identity."""

    def __init__(self, base_model, skip_layers: set):
        super().__init__()
        self.base = base_model
        self.skip_layers = set(skip_layers)

    @property
    def config(self):
        return self.base.config

    @property
    def vocab_size(self):
        return self.base.config.vocab_size

    def forward(self, input_ids, labels=None, attention_mask=None, cu_seqlens=None, **kwargs):
        base = self.base.model  # TransformerModel
        h = base.embeddings(input_ids)
        for layer_idx, layer in enumerate(base.layers):
            if layer_idx in self.skip_layers:
                continue  # identity
            outs = layer(
                hidden_states=h,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )
            h = outs[0]
        h = base.norm(h)
        logits = self.base.lm_head(h)
        loss = None
        if labels is not None:
            shifted = torch.cat([labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), shifted.view(-1), ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


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
        total_loss += float(out["loss"]) * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


def finetune_recovery(model, tok, device, tokens, lr, seq_len, log_every, eval_every,
                       seed=42, output_dir=None):
    """Brief fine-tune of the pruned model to recover lost quality."""
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=0.0,
    )
    max_steps = max(tokens // seq_len, 1)
    warmup = max(max_steps // 20, 50)

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        progress = min(progress, 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    train_dl = build_text_dataloader(
        tokenizer=tok, dataset=DATA_PATH, dataset_name=None, dataset_split="train",
        data_dir=None, data_files=None, seq_len=seq_len, context_len=2048,
        batch_size=1, num_workers=4, streaming=True, varlen=True, seed=seed,
    )
    train_iter = iter(train_dl)
    loss_ema = None
    t0 = time.time()
    log_f = open(Path(output_dir) / "recovery.log", "a") if output_dir else None

    for step in range(1, max_steps + 1):
        batch = next(train_iter)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        cu = batch.get("cu_seqlens")
        if cu is not None:
            cu = cu.to(device)
        out = model(input_ids=ids, labels=labels, cu_seqlens=cu)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scale = lr_scale(step)
        for g in opt.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * scale
        opt.step()

        lf = float(loss.detach())
        loss_ema = lf if loss_ema is None else 0.98 * loss_ema + 0.02 * lf
        if step % log_every == 0 or step == 1 or step == max_steps:
            tps = step * seq_len / (time.time() - t0)
            line = f"recover step {step}/{max_steps} loss={loss_ema:.3f} lr_scale={scale:.3f} tps={tps:,.0f}"
            print(f"[recov] {line}")
            if log_f:
                log_f.write(line + "\n")
                log_f.flush()
        if step % eval_every == 0 or step == max_steps:
            ppl = measure_ppl(model, tok, device, num_batches=4)
            line = f"recover eval step={step} ppl={ppl:.3f}"
            print(f"[recov] {line}")
            if log_f:
                log_f.write(line + "\n")
                log_f.flush()
            model.train()

    if log_f:
        log_f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-layers", type=str, default="8,9",
                   help="comma-separated layer indices to skip (block 4 = layers 8,9 at 2-per-block)")
    p.add_argument("--recover-tokens", type=int, default=100_000_000)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--output-dir", type=str, required=True)
    args = p.parse_args()

    skip_layers = set(int(x) for x in args.skip_layers.split(","))
    print(f"Pruning layers: {sorted(skip_layers)}")

    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=dtype,
    ).to(device)

    pruned = PrunedTransformer(base, skip_layers=skip_layers).to(device=device, dtype=dtype)

    # Initial PPL (before recovery)
    init_ppl = measure_ppl(pruned, tok, device, num_batches=8)
    print(f"[prune] Initial PPL (after pruning, no recovery): {init_ppl:.3f}")

    # Recovery fine-tune
    print(f"[prune] Starting recovery fine-tune ({args.recover_tokens:,} tokens)")
    finetune_recovery(
        pruned, tok, device, tokens=args.recover_tokens, lr=args.lr,
        seq_len=args.seq_len, log_every=args.log_every, eval_every=args.eval_every,
        output_dir=args.output_dir,
    )

    # Final PPL
    final_ppl = measure_ppl(pruned, tok, device, num_batches=8)
    print(f"[prune] FINAL: init_ppl={init_ppl:.3f}, final_ppl={final_ppl:.3f}, "
          f"pruned {len(skip_layers)}/{pruned.config.num_hidden_layers} layers")

    with open(Path(args.output_dir) / "summary.txt", "w") as f:
        f.write(f"Skipped layers: {sorted(skip_layers)}\n")
        f.write(f"Init PPL (after pruning, no recovery): {init_ppl:.3f}\n")
        f.write(f"Final PPL (after {args.recover_tokens:,} tokens recovery): {final_ppl:.3f}\n")
        f.write(f"Pruned {len(skip_layers)}/{pruned.config.num_hidden_layers} layers\n")


if __name__ == "__main__":
    main()
