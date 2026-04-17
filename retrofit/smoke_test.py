"""Smoke test for retrofit routes A/B/C.

Loads pretrained transformer_test (74M standard transformer), wraps with
each retrofit route, measures initial fineweb PPL, and confirms training
step doesn't explode.

Pass criteria:
  - Route A/C: PPL at init ≤ 1.1 × original (should be much closer; A
    starts beta=0 so exactly original; C has informed init, so close)
  - Route B: forward unchanged → PPL EXACTLY original
  - Train step: loss finite, not NaN
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flash-linear-attention")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/flame")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/experiments")

import fla  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from flame_reskip_common import build_text_dataloader, count_valid_tokens  # noqa: E402

from retrofit_model import RetrofitModel  # noqa: E402


MODEL_PATH = "/home/user01/Minko/reskip2/reskip/flame/saves/transformer_test"
DATA_PATH = "/home/user01/Minko/datasets/fineweb_edu_100BT"


def load_base(device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=dtype
    ).to(device)
    return base, tok


@torch.no_grad()
def measure_ppl(
    model,
    tok,
    device: str,
    num_batches: int = 16,
    is_retrofit: bool = False,
    seq_len: int = 65536,
):
    model.eval()
    dl = build_text_dataloader(
        tokenizer=tok,
        dataset=DATA_PATH,
        dataset_name=None,
        dataset_split="train",
        data_dir=None,
        data_files=None,
        seq_len=seq_len,
        context_len=2048,
        batch_size=1,
        num_workers=2,
        streaming=True,
        varlen=True,
        seed=0,
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
        kw = {}
        if cu is not None:
            kw["cu_seqlens"] = cu
        if is_retrofit:
            out = model(input_ids=ids, labels=labels, **kw)
            loss = out.loss
        else:
            out = model(input_ids=ids, labels=labels, use_cache=False, **kw)
            loss = out.loss
        n = count_valid_tokens(labels)
        total_loss += float(loss) * n
        total_tokens += n
    avg = total_loss / total_tokens
    return math.exp(avg), avg, total_tokens


def get_tiny_calibration_batches(tok, num_batches: int = 4, seq_len: int = 8192):
    """Grab a few fineweb batches for informed-init calibration."""
    dl = build_text_dataloader(
        tokenizer=tok,
        dataset=DATA_PATH,
        dataset_name=None,
        dataset_split="train",
        data_dir=None,
        data_files=None,
        seq_len=seq_len,
        context_len=2048,
        batch_size=1,
        num_workers=2,
        streaming=True,
        varlen=True,
        seed=1,  # different seed from eval
    )
    batches = []
    for i, batch in enumerate(dl):
        if i >= num_batches:
            break
        batches.append(batch["input_ids"])
    return batches


def train_step_check(model, tok, device: str):
    """Run a single fwd+bwd pass; verify loss is finite."""
    model.train()
    ids = torch.randint(0, tok.vocab_size, (1, 512), device=device)
    out = model(input_ids=ids, labels=ids)
    loss = out.loss
    if out.distill_loss is not None:
        # Route B combines
        total = loss + 0.1 * out.distill_loss
    else:
        total = loss
    total.backward()
    finite = torch.isfinite(total).item()
    # Check any grad is finite
    any_grad = False
    any_nan = False
    for p in model.trainable_parameters():
        if p.grad is not None:
            any_grad = True
            if not torch.isfinite(p.grad).all():
                any_nan = True
    return {
        "loss": float(loss.detach()),
        "distill_loss": float(out.distill_loss.detach()) if out.distill_loss is not None else None,
        "total": float(total.detach()),
        "finite": finite,
        "any_grad": any_grad,
        "any_nan_grad": any_nan,
    }


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    num_eval_batches = int(os.environ.get("NUM_BATCHES", "8"))

    print(f"[smoke] Loading baseline model from {MODEL_PATH}")
    base, tok = load_base(device=device, dtype=dtype)

    print("[smoke] Measuring baseline PPL...")
    base_ppl, base_loss, n_tok = measure_ppl(
        base, tok, device=device, num_batches=num_eval_batches, is_retrofit=False
    )
    print(f"[smoke] BASELINE: ppl={base_ppl:.3f}, loss={base_loss:.4f}, tokens={n_tok}")

    for route in ["A", "B", "C"]:
        print(f"\n{'=' * 60}")
        print(f"[smoke] Testing Route {route}")
        print(f"{'=' * 60}")

        # Reload base to ensure clean weights
        base2, _ = load_base(device=device, dtype=dtype)
        model = RetrofitModel(base2, num_blocks=6, route=route).to(device=device, dtype=dtype)

        # For Route C, apply informed init using calibration data
        if route == "C":
            print("[smoke-C] Collecting calibration keys for informed init...")
            cal_batches = get_tiny_calibration_batches(tok, num_batches=2, seq_len=2048)
            keys = model.collect_block_keys_for_init(cal_batches, device=device)
            # keys[i] corresponds to source-position i. The router has
            # positions 0..num_blocks; we init w_n (n>=1) with k_{n-1}.
            # So pass keys[0..num_blocks-1] in order; router.set_informed_init
            # will put keys[n-1] into w_n.
            # scale=10 further sharpens softmax concentration on n-1
            model.router.set_informed_init(keys, scale=10.0)
            # Set initial temperature low (sharp softmax) so α ≈ one-hot at n-1
            with torch.no_grad():
                model.router.log_tau_multiplier.fill_(math.log(0.1))
            print(f"[smoke-C] Informed init applied: collected {len(keys)} mean keys")

        # Cast new params to bf16 as well
        if route == "A":
            model.gate_logits.data = model.gate_logits.data.to(dtype)

        # Initial PPL
        init_ppl, init_loss, _ = measure_ppl(
            model, tok, device=device, num_batches=num_eval_batches, is_retrofit=True
        )
        ratio = init_ppl / base_ppl
        print(f"[smoke-{route}] Init PPL={init_ppl:.3f} (baseline {base_ppl:.3f}, ratio {ratio:.3f})")
        pass_ppl = ratio <= 1.10 if route != "B" else ratio <= 1.01
        print(f"[smoke-{route}] PPL preservation {'PASS' if pass_ppl else 'FAIL'}")

        # Training step
        train_info = train_step_check(model, tok, device=device)
        print(f"[smoke-{route}] Train step: {train_info}")
        pass_train = train_info["finite"] and train_info["any_grad"] and not train_info["any_nan_grad"]
        print(f"[smoke-{route}] Training stability {'PASS' if pass_train else 'FAIL'}")

        # Summary
        print(f"[smoke-{route}] OVERALL: {'PASS' if pass_ppl and pass_train else 'FAIL'}")

        del base2, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
