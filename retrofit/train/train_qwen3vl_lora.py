"""LoRA baseline: train base Qwen3-VL-2B with same data, same steps, similar
param budget as our retrofit. Fair apples-to-apples comparison.

- Freeze base; add LoRA on attention q_proj, v_proj of all 28 text decoder layers
- rank=32 → ~7.34M trainable params, close to retrofit's 7.4M
- Same data mix (UltraChat + LLaVA), same loss (assistant-masked CE only)
- No skip-branch KL, no entropy reg, no γ/adapter — pure LoRA SFT
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model

from train_qwen3vl_attnres_retrofit import (
    compute_assistant_mask, ultrachat_iter, llava_iter, mixed_stream, move_inputs,
    MODEL_PATH,
)


def train(args):
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[lora] Loading base from {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)

    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Apply LoRA on attention q_proj, v_proj of language decoder layers
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=args.target_modules.split(","),
        task_type=None,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Collect trainable parameters (LoRA A/B)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable: {n_train/1e6:.2f}M / total: {n_total/1e9:.2f}B  ({n_train/n_total*100:.3f}%)")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    max_steps = args.steps
    warmup = args.warmup_steps

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(
            math.pi * min((step - warmup) / max(max_steps - warmup, 1), 1.0)))

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a")

    stream = mixed_stream(processor, seed=args.seed, p_mm=args.p_multimodal)
    t0 = time.time()
    ce_ema = None

    for step in range(1, max_steps + 1):
        while True:
            inputs, labels, kind = next(stream)
            if inputs["input_ids"].shape[1] <= args.max_seq:
                break
        inputs = move_inputs(inputs, device)
        labels = labels.unsqueeze(0).to(device)

        model.train()
        out = model(**inputs, use_cache=False)
        logits = out.logits
        # shift for next-token CE on assistant-masked labels
        shifted = torch.cat(
            [labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1
        )
        V = logits.shape[-1]
        ce_loss = F.cross_entropy(
            logits.view(-1, V), shifted.view(-1), ignore_index=-100,
        )

        opt.zero_grad()
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scale = lr_scale(step)
        for g in opt.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * scale
        opt.step()

        cf = float(ce_loss.detach())
        ce_ema = cf if ce_ema is None else 0.98 * ce_ema + 0.02 * cf

        if step % args.log_every == 0 or step == 1 or step == max_steps:
            elapsed = time.time() - t0
            line = (f"step {step}/{max_steps} ce={ce_ema:.3f} kind={kind} "
                    f"lr_scale={scale:.3f} T_len={inputs['input_ids'].shape[1]} "
                    f"elapsed={elapsed:.0f}s")
            print(f"[lora] {line}", flush=True)
            log_f.write(line + "\n"); log_f.flush()

    # Save PEFT adapter
    model.save_pretrained(out_dir / "lora_adapter")
    print(f"[lora] saved {out_dir}/lora_adapter")
    log_f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--target-modules", default="q_proj,v_proj",
                   help="comma-separated module names for LoRA")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--max-seq", type=int, default=1536)
    p.add_argument("--p-multimodal", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
