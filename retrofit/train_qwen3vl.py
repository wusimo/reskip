"""Fine-tune Qwen3-VL-2B retrofit on text data.

Steps:
  1. Load Qwen3-VL-2B (full ~2B).
  2. Wrap with Qwen3VLRetrofit (Route A, β=0.1 init, positional bias).
  3. FREEZE base, train only router params (cheap retrofit).
  4. Fine-tune on fineweb_edu_100BT re-tokenized for Qwen tokenizer.
  5. Log PPL + α entropy.
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
from datasets import load_dataset

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoTokenizer

from qwen3vl_retrofit import Qwen3VLRetrofit


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"
DATA_PATH = "/home/user01/Minko/datasets/fineweb_edu_100BT"


def build_text_stream(tok, seq_len=2048, seed=0):
    """Stream fineweb text, tokenize with Qwen tokenizer, pack into fixed-length
    sequences. Returns an iterator of {input_ids, labels} tensors."""
    ds = load_dataset(DATA_PATH, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1000)

    buf = []
    eos_id = tok.eos_token_id
    if eos_id is None:
        eos_id = 151645  # Qwen end-of-text
    for sample in ds:
        text = sample.get("text", "")
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            chunk = buf[:seq_len]
            buf = buf[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)


@torch.no_grad()
def measure_ppl(model, tok, device, num_seqs=16, seq_len=2048):
    model.eval()
    stream = build_text_stream(tok, seq_len=seq_len, seed=0)
    total_loss = 0.0
    total_tokens = 0
    for i, seq in enumerate(stream):
        if i >= num_seqs:
            break
        ids = seq.unsqueeze(0).to(device)
        out = model(input_ids=ids, labels=ids)
        total_loss += float(out.loss) * seq_len
        total_tokens += seq_len
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def alpha_entropy(model, tok, device, num_seqs=2, seq_len=2048):
    model.eval()
    stream = build_text_stream(tok, seq_len=seq_len, seed=42)
    ent_accum = []
    for i, seq in enumerate(stream):
        if i >= num_seqs:
            break
        ids = seq.unsqueeze(0).to(device)
        out = model(input_ids=ids, labels=ids, return_alpha=True)
        for alpha in out.alpha_list:
            n = alpha.shape[-1]
            ent = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1).mean()
            ent_accum.append(float(ent) / math.log(n))
    return sum(ent_accum) / max(len(ent_accum), 1)


def train(args):
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    print(f"[qwen3vl-retrofit] Loading {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)

    print(f"[qwen3vl-retrofit] Wrapping with retrofit ({args.route}, num_blocks={args.num_blocks})")
    model = Qwen3VLRetrofit(
        base, num_blocks=args.num_blocks, route=args.route,
        beta_init_logit=args.beta_init_logit,
    ).to(device=device, dtype=dtype)
    if args.route == "A":
        model.gate_logits.data = model.gate_logits.data.to(dtype)

    if args.freeze_base:
        model.freeze_base()
        trainable = model.retrofit_parameters()
    else:
        trainable = model.trainable_parameters()

    n_trainable = sum(p.numel() for p in trainable)
    print(f"[qwen3vl-retrofit] Trainable params: {n_trainable / 1e6:.2f}M")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    max_steps = max(args.tokens // args.seq_len, 1)
    warmup = args.warmup_steps

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        progress = min(progress, 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a")

    # Initial stats
    init_ppl = measure_ppl(model, tok, device, num_seqs=8, seq_len=args.seq_len)
    init_ent = alpha_entropy(model, tok, device, num_seqs=2, seq_len=args.seq_len)
    msg = f"BEFORE ppl={init_ppl:.3f} alpha_ent={init_ent:.3f}"
    print(f"[qwen3vl-retrofit] {msg}")
    log_f.write(msg + "\n"); log_f.flush()

    # Train loop
    stream = build_text_stream(tok, seq_len=args.seq_len, seed=args.seed)
    t0 = time.time()
    loss_ema = None
    step = 0

    for step in range(1, max_steps + 1):
        try:
            seq = next(stream)
        except StopIteration:
            stream = build_text_stream(tok, seq_len=args.seq_len, seed=args.seed + step)
            seq = next(stream)
        ids = seq.unsqueeze(0).to(device)
        model.train()
        out = model(input_ids=ids, labels=ids)
        lm_loss = out.loss
        total = lm_loss
        if out.entropy_penalty is not None and args.entropy_weight > 0:
            total = total + args.entropy_weight * out.entropy_penalty

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scale = lr_scale(step)
        for g in opt.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * scale
        opt.step()

        lf = float(lm_loss.detach())
        loss_ema = lf if loss_ema is None else 0.98 * loss_ema + 0.02 * lf

        if step % args.log_every == 0 or step == 1 or step == max_steps:
            tps = step * args.seq_len / max(time.time() - t0, 1e-6)
            line = (f"step {step}/{max_steps} lm_loss={loss_ema:.3f} "
                    f"lr_scale={scale:.3f} tps={tps:,.0f}")
            if args.route == "A":
                betas = torch.sigmoid(model.gate_logits).detach().cpu().tolist()
                line += f" β=[{betas[0]:.3f}..{betas[-1]:.3f}]"
            print(f"[qwen3vl-retrofit] {line}")
            log_f.write(line + "\n"); log_f.flush()

        if step % args.eval_every == 0 or step == max_steps:
            ppl = measure_ppl(model, tok, device, num_seqs=4, seq_len=args.seq_len)
            ent = alpha_entropy(model, tok, device, num_seqs=2, seq_len=args.seq_len)
            line = f"eval step={step} ppl={ppl:.3f} alpha_ent={ent:.3f}"
            print(f"[qwen3vl-retrofit] {line}")
            log_f.write(line + "\n"); log_f.flush()

    # Final eval
    final_ppl = measure_ppl(model, tok, device, num_seqs=16, seq_len=args.seq_len)
    final_ent = alpha_entropy(model, tok, device, num_seqs=4, seq_len=args.seq_len)
    msg = (f"FINAL init_ppl={init_ppl:.3f} final_ppl={final_ppl:.3f} "
           f"init_ent={init_ent:.3f} final_ent={final_ent:.3f}")
    print(f"[qwen3vl-retrofit] {msg}")
    log_f.write(msg + "\n"); log_f.close()

    # Save router state dict
    state = {
        "router": model.router.state_dict(),
        "config": {
            "route": args.route,
            "num_blocks": args.num_blocks,
            "tokens": step * args.seq_len,
            "init_ppl": init_ppl,
            "final_ppl": final_ppl,
            "final_ent": final_ent,
        },
    }
    if args.route == "A":
        state["gate_logits"] = model.gate_logits.data.cpu()
    torch.save(state, out_dir / "retrofit_state.pt")
    print(f"[qwen3vl-retrofit] Saved {out_dir}/retrofit_state.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--route", choices=["A", "B"], default="A")
    p.add_argument("--num-blocks", type=int, default=14)
    p.add_argument("--tokens", type=int, default=100_000_000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--beta-init-logit", type=float, default=-2.2)
    p.add_argument("--entropy-weight", type=float, default=0.05)
    p.add_argument("--freeze-base", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
