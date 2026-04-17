"""Fine-tune a retrofitted standard transformer on fineweb.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_retrofit.py --route A --tokens 500_000_000 \
      --output-dir outputs/retrofit_A_500M

Route A: trains pseudo-queries + gate β + (optionally) base weights
Route B: trains pseudo-queries + adds distillation loss; base weights can
         be frozen (default) since forward is unchanged
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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


@torch.no_grad()
def measure_ppl(model, tok, device, num_batches=8, seq_len=65536):
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
    avg = total_loss / total_tokens
    return math.exp(avg), avg


@torch.no_grad()
def alpha_entropy(model, tok, device, num_batches=2, seq_len=8192):
    """Compute average per-token alpha entropy across positions."""
    model.eval()
    dl = build_text_dataloader(
        tokenizer=tok, dataset=DATA_PATH, dataset_name=None, dataset_split="train",
        data_dir=None, data_files=None, seq_len=seq_len, context_len=2048,
        batch_size=1, num_workers=2, streaming=True, varlen=True, seed=42,
    )
    ent_accum = []
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
        for alpha in out.alpha_list:
            # alpha: [B, T, n]  — per-token distribution
            n = alpha.shape[-1]
            ent = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1)
            norm_ent = ent.mean().item() / math.log(n)
            ent_accum.append(norm_ent)
    return sum(ent_accum) / max(len(ent_accum), 1)


def train(args):
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[train] Loading base model from {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=dtype
    ).to(device)

    model = RetrofitModel(
        base, num_blocks=args.num_blocks, route=args.route
    ).to(device=device, dtype=dtype)

    if args.route == "A":
        model.gate_logits.data = model.gate_logits.data.to(dtype)

    # Freeze policy
    if args.freeze_base:
        model.freeze_base()

    # Optim: separate param groups for base vs new
    base_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("base_model.")]
    new_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith("base_model.")]
    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr_base})
    if new_params:
        param_groups.append({"params": new_params, "lr": args.lr_new})
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.0)

    # Simple linear warmup + cosine decay
    warmup = args.warmup_steps
    max_steps = args.max_steps

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        progress = min(progress, 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    # Before-training ppl + alpha entropy
    init_ppl, init_loss = measure_ppl(model, tok, device, num_batches=4)
    init_ent = alpha_entropy(model, tok, device, num_batches=2)
    print(f"[train] BEFORE: ppl={init_ppl:.3f}, alpha_ent={init_ent:.3f}")

    # Training dataloader (streaming, different seed from eval)
    train_dl = build_text_dataloader(
        tokenizer=tok, dataset=DATA_PATH, dataset_name=None, dataset_split="train",
        data_dir=None, data_files=None, seq_len=args.seq_len, context_len=2048,
        batch_size=1, num_workers=4, streaming=True, varlen=True, seed=args.seed,
    )
    train_iter = iter(train_dl)

    step = 0
    t0 = time.time()
    loss_ema = None
    distill_ema = None

    # β curriculum schedule (only active if args.beta_schedule and route == "A")
    def beta_logit_at(step):
        """Ramp β_logit from start (warmup_frac * steps) to end (ramp_end_frac * steps)."""
        warmup_end = int(args.beta_warmup_frac * max_steps)
        ramp_end = int(args.beta_ramp_end_frac * max_steps)
        start_logit = args.beta_start_logit
        end_logit = args.beta_end_logit
        if step < warmup_end:
            return start_logit
        if step >= ramp_end:
            return end_logit
        progress = (step - warmup_end) / max(ramp_end - warmup_end, 1)
        return start_logit + progress * (end_logit - start_logit)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_f = open(log_path, "a")

    while step < max_steps:
        model.train()
        batch = next(train_iter)
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        cu = batch.get("cu_seqlens")
        if cu is not None:
            cu = cu.to(device)
        # β-curriculum override (Route A only, when enabled)
        if args.beta_schedule and args.route == "A":
            bl = beta_logit_at(step)
            with torch.no_grad():
                model.gate_logits.data.fill_(bl)
        out = model(input_ids=ids, labels=labels, cu_seqlens=cu)
        lm_loss = out.loss
        total_loss = lm_loss
        if out.distill_loss is not None:
            total_loss = total_loss + args.distill_weight * out.distill_loss
        if out.entropy_penalty is not None and args.entropy_weight > 0:
            # entropy_penalty is the α entropy; we want it LOW (sparse α) so ADD it
            total_loss = total_loss + args.entropy_weight * out.entropy_penalty

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        # Apply lr schedule
        scale = lr_scale(step)
        for g in opt.param_groups:
            g["lr"] = g.get("base_lr", g["lr"])
            # Store base_lr on first use
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * scale
        opt.step()

        lm_f = float(lm_loss.detach())
        loss_ema = lm_f if loss_ema is None else 0.98 * loss_ema + 0.02 * lm_f
        if out.distill_loss is not None:
            d_f = float(out.distill_loss.detach())
            distill_ema = d_f if distill_ema is None else 0.98 * distill_ema + 0.02 * d_f

        step += 1
        tokens_seen = step * args.seq_len * args.batch_size
        if step % args.log_every == 0 or step == 1 or step == max_steps:
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed
            line = (f"step {step}/{max_steps} tokens={tokens_seen:,} "
                    f"lm_loss={loss_ema:.3f}")
            if distill_ema is not None:
                line += f" distill={distill_ema:.3f}"
            line += f" lr_scale={scale:.3f} tps={tps:,.0f}"
            if args.route == "A":
                betas = torch.sigmoid(model.gate_logits).detach().cpu().tolist()
                line += f" β={[f'{b:.3f}' for b in betas]}"
            print(f"[train] {line}")
            log_f.write(line + "\n")
            log_f.flush()

        if step % args.eval_every == 0 or step == max_steps:
            eval_ppl, eval_loss = measure_ppl(model, tok, device, num_batches=4)
            eval_ent = alpha_entropy(model, tok, device, num_batches=2)
            line = f"eval step={step} ppl={eval_ppl:.3f} loss={eval_loss:.4f} alpha_ent={eval_ent:.3f}"
            print(f"[train] {line}")
            log_f.write(line + "\n")
            log_f.flush()

    # Final evaluation
    final_ppl, final_loss = measure_ppl(model, tok, device, num_batches=8)
    final_ent = alpha_entropy(model, tok, device, num_batches=4)
    line = (f"FINAL route={args.route} init_ppl={init_ppl:.3f} "
            f"final_ppl={final_ppl:.3f} init_ent={init_ent:.3f} "
            f"final_ent={final_ent:.3f}")
    print(f"[train] {line}")
    log_f.write(line + "\n")
    log_f.close()

    # Save model state dict (just the retrofit-added params for lightness)
    state = {
        "router": model.router.state_dict(),
        "config": {
            "route": args.route,
            "num_blocks": args.num_blocks,
            "seq_len": args.seq_len,
            "tokens": tokens_seen,
            "init_ppl": init_ppl,
            "final_ppl": final_ppl,
            "final_ent": final_ent,
        },
    }
    if args.route == "A":
        state["gate_logits"] = model.gate_logits.data.cpu()
    if not args.freeze_base:
        state["base_state_dict"] = model.base_model.state_dict()
    torch.save(state, out_dir / "retrofit_state.pt")
    print(f"[train] Saved to {out_dir}/retrofit_state.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", choices=["A", "B"], required=True)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--tokens", type=int, default=500_000_000, help="total training tokens")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr-base", type=float, default=1e-5, help="LR for pretrained base params")
    p.add_argument("--lr-new", type=float, default=3e-4, help="LR for retrofit-added params")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--distill-weight", type=float, default=0.1)
    p.add_argument("--entropy-weight", type=float, default=0.0,
                   help="coefficient on α entropy (encourages sparse routing)")
    p.add_argument("--freeze-base", action="store_true", default=False)
    p.add_argument("--beta-schedule", action="store_true",
                   help="Route A only: externally schedule β_logit to force β→1 during training")
    p.add_argument("--beta-start-logit", type=float, default=-2.2,
                   help="β_logit at start of ramp (default -2.2 → β=0.10)")
    p.add_argument("--beta-end-logit", type=float, default=4.0,
                   help="β_logit at end of ramp (default 4.0 → β=0.982)")
    p.add_argument("--beta-warmup-frac", type=float, default=0.2,
                   help="fraction of training to hold β at start")
    p.add_argument("--beta-ramp-end-frac", type=float, default=0.8,
                   help="fraction of training by which β reaches end value")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", type=str, required=True)
    args = p.parse_args()

    # Compute max_steps from tokens
    args.max_steps = max(args.tokens // (args.seq_len * args.batch_size), 1)
    print(f"[train] max_steps={args.max_steps} ({args.tokens} tokens / "
          f"{args.seq_len * args.batch_size} per step)")
    train(args)


if __name__ == "__main__":
    main()
