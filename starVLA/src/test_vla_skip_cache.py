"""Smoke test: VLA StarVLABackboneSkipContext with KV cache + dynamic skip.

Builds a StarVLAAttnResAdapter, monkey-patches a fresh Qwen3-VL-2B's
language_model via the context, runs four combinations:

  (1) no-skip + use_cache=False   — baseline; tests plain in-backbone AttnRes.
  (2) no-skip + use_cache=True    — tests cache population through the patch.
  (3) skip via dyn_cfg + use_cache=True — exercises the K/V-only skip path.
  (4) skip via dyn_cfg + use_cache=False — parity check against (3).

Loads H_r256_5k warm-start state when available so the router/adapter/γ
are non-trivial. Asserts next-token argmax matches between (1)/(2) and
(3)/(4) respectively — i.e., use_cache flip must not change the argmax
at matching skip config (cache path is numerically equivalent).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/starVLA/src")
from starvla_integration import StarVLAAttnResAdapter, StarVLABackboneSkipContext

MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"
STATE_PATH = "/home/user01/Minko/reskip2/reskip/retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt"


def run_once(base, adapter, ids, *, enable_skipping, dynamic_skip_config, use_cache):
    ctx = StarVLABackboneSkipContext(
        adapter,
        enable_skipping=enable_skipping,
        dynamic_skip_config=dynamic_skip_config,
    )
    ctx.bind_text_model(base.model.language_model)
    with ctx, torch.no_grad():
        out = base(input_ids=ids, use_cache=use_cache)
    return out, ctx.get_summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--state-path", default=STATE_PATH)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    num_hidden_layers = base.config.text_config.num_hidden_layers
    hidden_size = base.config.text_config.hidden_size

    adapter = StarVLAAttnResAdapter(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        n_blocks=14,
        adapter_rank=256,
    ).to(device=device, dtype=dtype).eval()
    if os.path.exists(args.state_path):
        adapter.load_retrofit_state(args.state_path, strict=False)
        adapter = adapter.to(device=device, dtype=dtype)
        print(f"[init] loaded H_r256_5k state from {args.state_path}")
    else:
        print(f"[init] state missing at {args.state_path}, using random init (γ=1 default)")

    ids = tok(
        "The capital of France is Paris. The capital of Germany is",
        return_tensors="pt",
    ).input_ids.to(device)

    # (1) no-skip + use_cache=False
    out1, s1 = run_once(base, adapter, ids,
                        enable_skipping=False, dynamic_skip_config=None, use_cache=False)
    # (2) no-skip + use_cache=True
    out2, s2 = run_once(base, adapter, ids,
                        enable_skipping=False, dynamic_skip_config=None, use_cache=True)
    # Synthetic dynamic skip config — intentionally low thresholds so blocks will fire.
    # Uses retrofit's schema: per-block τ + eligible + max_skips.
    dyn_cfg = {
        "thresholds": {b: 0.0 for b in range(14)},  # τ=0 → always trigger where eligible
        "eligible_blocks": {4, 6, 11},
        "max_skips": 2,
    }
    # (3) skip + use_cache=True
    out3, s3 = run_once(base, adapter, ids,
                        enable_skipping=True, dynamic_skip_config=dyn_cfg, use_cache=True)
    # (4) skip + use_cache=False
    out4, s4 = run_once(base, adapter, ids,
                        enable_skipping=True, dynamic_skip_config=dyn_cfg, use_cache=False)

    def next_tok(out):
        return int(out.logits[0, -1].argmax())

    def max_diff(a, b):
        return float((a.float() - b.float()).abs().max())

    print(f"\n=== forward shapes and summaries ===")
    for i, (out, s) in enumerate([(out1, s1), (out2, s2), (out3, s3), (out4, s4)], start=1):
        print(f"  ({i}) logits={tuple(out.logits.shape)}  summary={s}")

    print(f"\n=== next-token argmax ===")
    t1, t2, t3, t4 = next_tok(out1), next_tok(out2), next_tok(out3), next_tok(out4)
    print(f"  (1) no-skip, use_cache=False → {tok.decode([t1])!r}")
    print(f"  (2) no-skip, use_cache=True  → {tok.decode([t2])!r}")
    print(f"  (3) skip,    use_cache=True  → {tok.decode([t3])!r}")
    print(f"  (4) skip,    use_cache=False → {tok.decode([t4])!r}")
    print(f"\n=== last-position logit jitter across use_cache flip ===")
    print(f"  no-skip: (1) vs (2) max|Δ| = {max_diff(out1.logits[:, -1], out2.logits[:, -1]):.4f}")
    print(f"  skip:    (3) vs (4) max|Δ| = {max_diff(out3.logits[:, -1], out4.logits[:, -1]):.4f}")

    assert t1 == t2, (
        "use_cache flag changed argmax under no-skip — patched forward's cache path is wrong"
    )
    assert t3 == t4, (
        "use_cache flag changed argmax under skip — K/V-only skip cache path is wrong"
    )
    assert s3["skipped_blocks"], "dyn_cfg with τ=0 should have triggered at least one skip"
    assert s3["skipped_blocks"] == s4["skipped_blocks"], (
        "same cfg produced different skip decisions under use_cache flip — state leakage bug"
    )

    print(f"\n✓ all four combinations agree on next-token argmax; skipped_blocks={s3['skipped_blocks']}")


if __name__ == "__main__":
    main()
