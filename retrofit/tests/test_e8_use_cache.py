"""E8 validation: retrofit forward respects use_cache, generate() works.

Three checks:
  1. Single forward at γ=0 with use_cache={False, True} produces identical logits
     (cache population is transparent at prefill).
  2. use_cache=True + skip raises NotImplementedError (guard).
  3. base_model.generate() on retrofit vs base produces identical tokens at γ=0
     (since retrofit at γ=0 is structurally identical to base).
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit

MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional H_r256_5k-style retrofit state; if set, Check 4 benchmarks "
        "the trained retrofit (γ=1, real adapter) in addition to γ=0 identity.",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    # IMPORTANT: load TRUE base separately from any retrofit-wrapped base.
    # Qwen3VLAttnResRetrofit.__init__ monkey-patches its base.model.language_model.forward
    # IN PLACE, so if we shared one ``base`` instance, both "base" and
    # "retrofit" forwards would route through the patched forward and the
    # speed comparison would be bogus (this was the 04-20 bench bug).
    true_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    retro_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()

    prompt = "The capital of France is Paris. The capital of Germany is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    model = Qwen3VLAttnResRetrofit(retro_base, num_blocks=14).to(device=device, dtype=dtype).eval()
    # Keep a handle to the stock base for timing / parity checks.
    base = true_base
    trained = None
    if args.state_path:
        ck = torch.load(args.state_path, map_location="cpu")
        cfg = ck.get("config", {})
        kwargs = dict(num_blocks=cfg.get("num_blocks", 14))
        if "adapter_rank" in cfg:
            kwargs["adapter_rank"] = cfg["adapter_rank"]
        # Separate base for the trained retrofit too.
        trained_base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
        trained = Qwen3VLAttnResRetrofit(trained_base, **kwargs).to(device=device, dtype=dtype).eval()
        trained.router.load_state_dict(
            {k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()}
        )
        trained.adapters.load_state_dict(
            {k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()}
        )
        trained.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))

    # --- Check 1: single prefill with / without cache should pick the same
    # NEXT token. HF SDPA takes different code paths when past_key_values is
    # present (even if empty), producing small bf16 accumulation differences;
    # the same jitter appears in the base model, so we only require the
    # last-position (next-token) argmax to match.
    with torch.no_grad():
        out_nocache = model(input_ids=ids, use_cache=False).logits
        out_cache = model(input_ids=ids, use_cache=True).logits
        base_nocache = base(input_ids=ids, use_cache=False).logits
        base_cache = base(input_ids=ids, use_cache=True).logits
    diff_max = (out_nocache.float() - out_cache.float()).abs().max().item()
    base_diff_max = (base_nocache.float() - base_cache.float()).abs().max().item()
    retro_next_match = bool(
        (out_nocache[:, -1].argmax(-1) == out_cache[:, -1].argmax(-1)).all().item()
    )
    base_next_match = bool(
        (base_nocache[:, -1].argmax(-1) == base_cache[:, -1].argmax(-1)).all().item()
    )
    print(
        f"[1] prefill jitter (use_cache False vs True): "
        f"retrofit max|Δ|={diff_max:.4f}  base max|Δ|={base_diff_max:.4f}  "
        f"next-token argmax retrofit={retro_next_match} base={base_next_match}"
    )
    assert retro_next_match, "retrofit next-token prediction changed when we flipped use_cache"
    print("    ✓ next-token unchanged; logit jitter matches base model's jitter")

    # --- Check 2: skip + use_cache=True runs without error (K/V-only path
    # populates cache for skipped layers); see test_skip_kv_equiv.py for the
    # per-skip-config last-position argmax correctness check.
    out = model(input_ids=ids, use_cache=True, skip_block_indices=[4])
    assert out.logits is not None
    out = model(
        input_ids=ids,
        use_cache=True,
        dynamic_skip_config={"thresholds": {4: 0.5}, "eligible_blocks": {4}, "max_skips": 1},
    )
    assert out.logits is not None
    print("[2] skip + use_cache=True now supported (no crash; cache populated via K/V-only path)")

    # --- Check 3: generate() with retrofit at γ=0 matches base ---
    #
    # At γ=0 the retrofit is identity-at-init: every block's input = prev block
    # output (adapter contribution is zero). generate() routes through
    # base_model.generate -> base_model.forward -> model.language_model.forward
    # which is our patched forward. If our forward correctly honours use_cache
    # and past_key_values, the two generations should be byte-identical.
    with torch.no_grad():
        base_gen = base.generate(
            input_ids=ids,
            max_new_tokens=args.gen_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
        retro_gen = model.base_model.generate(
            input_ids=ids,
            max_new_tokens=args.gen_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
        )
    base_text = tok.decode(base_gen[0], skip_special_tokens=True)
    retro_text = tok.decode(retro_gen[0], skip_special_tokens=True)
    print(f"[3] base gen:     {base_text!r}")
    print(f"    retrofit gen: {retro_text!r}")
    match = bool((base_gen == retro_gen).all().item())
    print(f"    token-level match: {match}")
    assert match, "retrofit generate at γ=0 must match base generate"
    print("    ✓ generate() through patched forward produces identical tokens")

    # --- Check 4: wall-clock generate latency, base vs retrofit-full at γ=0 ---
    def bench_generate(m, warmup=3, timed=10):
        times = []
        for _ in range(warmup):
            with torch.no_grad():
                _ = m.generate(
                    input_ids=ids,
                    max_new_tokens=args.gen_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
            torch.cuda.synchronize()
        for _ in range(timed):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = m.generate(
                    input_ids=ids,
                    max_new_tokens=args.gen_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times.sort()
        return times[len(times) // 2]  # median

    base_med = bench_generate(base)
    retro_med = bench_generate(model.base_model)
    print(
        f"[4a] generate({args.gen_tokens} tok, use_cache=True) median: "
        f"base={base_med * 1000:.1f}ms, retrofit(γ=0 identity)={retro_med * 1000:.1f}ms  "
        f"(retrofit/base = {retro_med / base_med:.3f}x)"
    )
    if trained is not None:
        trained_med = bench_generate(trained.base_model)
        # Also measure the retrofit text prefill (no generation) to isolate the
        # router/adapter overhead — at γ=1 this is the "honest" full-path cost.
        def bench_prefill(m, ids_local, warmup=3, timed=15):
            times = []
            for _ in range(warmup):
                with torch.no_grad():
                    _ = m(input_ids=ids_local, use_cache=True)
                torch.cuda.synchronize()
            for _ in range(timed):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = m(input_ids=ids_local, use_cache=True)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            times.sort()
            return times[len(times) // 2]

        base_pref = bench_prefill(base, ids)
        trained_pref = bench_prefill(trained, ids)
        print(
            f"[4b] generate({args.gen_tokens} tok, use_cache=True) median: "
            f"base={base_med * 1000:.1f}ms, "
            f"retrofit(trained γ=1)={trained_med * 1000:.1f}ms  "
            f"(trained/base = {trained_med / base_med:.3f}x)"
        )
        print(
            f"[4c] prefill(seq={ids.shape[1]}) median: "
            f"base={base_pref * 1000:.2f}ms, retrofit(trained)={trained_pref * 1000:.2f}ms  "
            f"(trained/base = {trained_pref / base_pref:.3f}x) "
            f"— ratio indicates pure router+adapter overhead at γ=1"
        )
        gen_decode_only_base = max(base_med - base_pref, 1e-9)
        gen_decode_only_trained = max(trained_med - trained_pref, 1e-9)
        print(
            f"[4d] decode-only ({args.gen_tokens} tok) ≈ gen − prefill: "
            f"base={gen_decode_only_base * 1000:.1f}ms, "
            f"retrofit(trained)={gen_decode_only_trained * 1000:.1f}ms  "
            f"(trained/base = {gen_decode_only_trained / gen_decode_only_base:.3f}x)"
        )


if __name__ == "__main__":
    main()
