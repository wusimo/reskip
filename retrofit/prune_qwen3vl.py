"""α-guided pruning of Qwen3-VL-2B text decoder + recovery fine-tune."""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoTokenizer
from train_qwen3vl import build_text_stream


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


class PrunedQwen3VL(nn.Module):
    """Wraps Qwen3-VL with specific text decoder layers replaced by identity."""

    def __init__(self, base_model, skip_layers: set):
        super().__init__()
        self.base = base_model
        self.skip_layers = set(skip_layers)

    @property
    def config(self):
        return self.base.config

    @property
    def vocab_size(self):
        return self.base.config.text_config.vocab_size

    def forward(self, input_ids, labels=None, attention_mask=None, position_ids=None, **kwargs):
        text_model = self.base.model.language_model
        h = text_model.embed_tokens(input_ids)
        B, T = input_ids.shape
        device = h.device
        if position_ids is None:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        position_embeddings = text_model.rotary_emb(h, position_ids)

        for layer_idx, layer in enumerate(text_model.layers):
            if layer_idx in self.skip_layers:
                continue
            out = layer(h, attention_mask=attention_mask, position_embeddings=position_embeddings, **kwargs)
            if isinstance(out, tuple):
                h = out[0]
            else:
                h = out

        h = text_model.norm(h)
        logits = self.base.lm_head(h)
        loss = None
        if labels is not None:
            shifted = torch.cat([labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), shifted.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}


@torch.no_grad()
def measure_ppl(model, tok, device, num_seqs=16, seq_len=2048, seed=0):
    model.eval()
    stream = build_text_stream(tok, seq_len=seq_len, seed=seed)
    total_loss = 0.0
    total_tokens = 0
    for i, seq in enumerate(stream):
        if i >= num_seqs:
            break
        ids = seq.unsqueeze(0).to(device)
        out = model(input_ids=ids, labels=ids)
        total_loss += float(out["loss"]) * seq_len
        total_tokens += seq_len
    return math.exp(total_loss / total_tokens)


def recovery_finetune(model, tok, device, tokens, lr, seq_len,
                      log_every, eval_every, seed, output_dir):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    max_steps = max(tokens // seq_len, 1)
    warmup = max(max_steps // 20, 50)

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(max_steps - warmup, 1)
        progress = min(progress, 1.0)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    log_f = open(Path(output_dir) / "recovery.log", "a")
    loss_ema = None
    t0 = time.time()
    stream = build_text_stream(tok, seq_len=seq_len, seed=seed)

    for step in range(1, max_steps + 1):
        try:
            seq = next(stream)
        except StopIteration:
            stream = build_text_stream(tok, seq_len=seq_len, seed=seed + step)
            seq = next(stream)
        ids = seq.unsqueeze(0).to(device)
        out = model(input_ids=ids, labels=ids)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scale = lr_scale(step)
        for g in opt.param_groups:
            if "base_lr" not in g:
                g["base_lr"] = g["lr"]
            g["lr"] = g["base_lr"] * scale
        opt.step()
        lf = float(loss.detach())
        loss_ema = lf if loss_ema is None else 0.98 * loss_ema + 0.02 * lf
        if step % log_every == 0 or step == 1 or step == max_steps:
            tps = step * seq_len / max(time.time() - t0, 1e-6)
            line = f"recover step {step}/{max_steps} loss={loss_ema:.3f} lr_scale={scale:.3f} tps={tps:,.0f}"
            print(f"[prune] {line}"); log_f.write(line + "\n"); log_f.flush()
        if step % eval_every == 0 or step == max_steps:
            ppl = measure_ppl(model, tok, device, num_seqs=4, seq_len=seq_len)
            line = f"recover eval step={step} ppl={ppl:.3f}"
            print(f"[prune] {line}"); log_f.write(line + "\n"); log_f.flush()
            model.train()
    log_f.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-layers", type=str, required=True,
                   help="comma-separated layer indices to skip")
    p.add_argument("--recover-tokens", type=int, default=100_000_000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--save-model", action="store_true",
                   help="save recovered base weights at end")
    args = p.parse_args()

    skip_layers = set(int(x) for x in args.skip_layers.split(","))
    print(f"[prune] Pruning layers: {sorted(skip_layers)}")

    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)

    pruned = PrunedQwen3VL(base, skip_layers=skip_layers).to(device=device, dtype=dtype)

    init_ppl = measure_ppl(pruned, tok, device, num_seqs=16, seq_len=args.seq_len)
    print(f"[prune] Initial PPL (after pruning, no recovery): {init_ppl:.3f}")

    if args.recover_tokens > 0:
        print(f"[prune] Recovery fine-tune: {args.recover_tokens:,} tokens, lr={args.lr}")
        recovery_finetune(
            pruned, tok, device,
            tokens=args.recover_tokens, lr=args.lr, seq_len=args.seq_len,
            log_every=args.log_every, eval_every=args.eval_every,
            seed=0, output_dir=args.output_dir,
        )
        final_ppl = measure_ppl(pruned, tok, device, num_seqs=16, seq_len=args.seq_len)
        print(f"[prune] FINAL PPL (after recovery): {final_ppl:.3f}")
    else:
        final_ppl = init_ppl

    with open(Path(args.output_dir) / "summary.txt", "w") as f:
        f.write(f"Skipped layers: {sorted(skip_layers)}\n")
        f.write(f"Init PPL (after pruning): {init_ppl:.3f}\n")
        f.write(f"Final PPL (after recovery): {final_ppl:.3f}\n")
        f.write(f"Pruned {len(skip_layers)}/{pruned.config.text_config.num_hidden_layers}\n")

    # Save recovered base weights + skip_layers list for later inference
    if args.save_model and args.recover_tokens > 0:
        state = {
            "skip_layers": sorted(skip_layers),
            "base_state_dict": pruned.base.state_dict(),
        }
        save_path = Path(args.output_dir) / "recovered_state.pt"
        torch.save(state, save_path)
        print(f"[prune] Saved recovered model to {save_path}")


if __name__ == "__main__":
    main()
