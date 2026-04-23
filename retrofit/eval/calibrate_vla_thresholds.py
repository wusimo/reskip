"""Produce a ``dynamic_skip_config`` JSON for VLA inference.

Calibrates per-block w_recent thresholds by running a Qwen3-VL-2B with the
VLA StarVLABackboneSkipContext patched on, mirroring retrofit's phase-1
calibration (``retrofit/eval_dynamic_skip.py``). The adapter weights can
come from:

  1. a retrofit state file (``retrofit/outputs/H_r256_5k/retrofit_attnres_state.pt``)
     — fastest path, matches the LLM retrofit's calibration exactly.
  2. a full VLA ckpt — extract ``attnres_adapter.*`` keys into a fresh
     ``StarVLAAttnResAdapter`` and calibrate against its router/adapter/γ.

Output JSON schema (matches what ``eval_libero.py --dyn-skip-config-path``
and ``StarVLABackboneSkipContext.dynamic_skip_config`` consume):

    {
      "thresholds": {"0": τ_0, "1": τ_1, ..., "13": τ_13},
      "eligible_blocks": [4, 6, 11] | null,
      "max_skips": 2 | null,
      "quantile": 0.85,
      "n_samples": 32,
      "source": "/abs/path/to/state_or_ckpt"
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/starVLA/src")
from starvla_integration import StarVLAAttnResAdapter, StarVLABackboneSkipContext

MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load_adapter_from_source(state_path: str, device, dtype, n_blocks=14, adapter_rank=256):
    """Build a StarVLAAttnResAdapter and load router/adapter/γ from either
    a retrofit state or a full VLA ckpt."""
    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    n_blocks = cfg.get("num_blocks", n_blocks)
    adapter_rank = cfg.get("adapter_rank", adapter_rank)

    # Detect layout.
    if isinstance(ck, dict) and "router" in ck and "adapters" in ck:
        # retrofit state format
        router_sd = ck["router"]
        adapters_sd = ck["adapters"]
        gamma = ck["gamma"]
    else:
        # Full state_dict from a VLA training run — keys start with
        # "attnres_adapter.router.*", "attnres_adapter.adapters.*",
        # "attnres_adapter.gamma".
        sd = ck if isinstance(ck, dict) else ck.state_dict()
        router_sd, adapters_sd, gamma = {}, {}, None
        for k, v in sd.items():
            if k.startswith("attnres_adapter.router."):
                router_sd[k.removeprefix("attnres_adapter.router.")] = v
            elif k.startswith("attnres_adapter.adapters."):
                adapters_sd[k.removeprefix("attnres_adapter.adapters.")] = v
            elif k == "attnres_adapter.gamma":
                gamma = v
        if not router_sd or gamma is None:
            raise ValueError(
                f"could not find attnres_adapter.* keys in {state_path}; "
                "expected either a retrofit state (router/adapters/gamma) or "
                "a VLA ckpt whose state_dict has attnres_adapter.* entries"
            )

    # Figure out hidden_size from router weights.
    hidden_size = router_sd["w_query"].shape[1]
    # Get num_hidden_layers from base model config at load time.
    base_cfg = AutoModelForImageTextToText.from_pretrained(MODEL_PATH).config.text_config
    num_hidden_layers = base_cfg.num_hidden_layers

    adapter = StarVLAAttnResAdapter(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        n_blocks=n_blocks,
        adapter_rank=adapter_rank,
    ).to(device=device, dtype=dtype).eval()
    adapter.router.load_state_dict(router_sd, strict=False)
    adapter.adapters.load_state_dict(adapters_sd, strict=False)
    adapter.gamma.data.copy_(gamma.to(device=device, dtype=dtype))
    return adapter


@torch.no_grad()
def collect_w_recent(base, adapter, tok, device, n_samples, seq_len):
    """Run N LAMBADA prefixes through the VLA in-backbone forward and
    collect the per-block w_recent values (alpha[..., -1].mean() per block).
    """
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    ds = ds.select(range(n_samples, n_samples + n_samples))
    per_block = defaultdict(list)
    ctx = StarVLABackboneSkipContext(adapter, enable_skipping=False, dynamic_skip_config=None)
    ctx.bind_text_model(base.model.language_model)
    for ex in ds:
        ids = tok.encode(ex["text"].strip(), add_special_tokens=False)[:seq_len]
        if not ids:
            continue
        inp = torch.tensor([ids], device=device)
        # Re-enter ctx each iteration so patch/unpatch is clean; bind is sticky.
        with ctx:
            # Patched forward stores alpha on ctx.adapter only implicitly via
            # the router call; we need to grab alpha_list. Peek via the
            # summary's routing — but the summary doesn't carry per-block
            # alphas. Instead, hijack the router.route method to record the
            # mean w_recent as it runs. Simpler: monkey-patch route for this call.
            alphas: list[float] = []
            orig_route = adapter.router.route

            def recording_route(position, completed, _alphas=alphas, _orig=orig_route):
                routed, alpha = _orig(position, completed)
                if alpha.shape[-1] >= 2:
                    _alphas.append(float(alpha[..., -1].float().mean().item()))
                else:
                    _alphas.append(float("nan"))
                return routed, alpha

            adapter.router.route = recording_route
            try:
                _ = base(input_ids=inp, use_cache=False)
            finally:
                adapter.router.route = orig_route
            for b, w in enumerate(alphas):
                per_block[b].append(w)
    return dict(per_block)


def pick_thresholds(per_block, quantile):
    out = {}
    for b, vals in per_block.items():
        vals = [v for v in vals if v == v]  # drop nan
        if not vals:
            continue
        vs = sorted(vals)
        out[int(b)] = float(vs[int(quantile * (len(vs) - 1))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-path", required=True,
                    help="retrofit state (router/adapters/gamma) OR a VLA ckpt whose state_dict holds attnres_adapter.*")
    ap.add_argument("--output", required=True, help="JSON output path")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--quantile", type=float, default=0.85)
    ap.add_argument("--eligible", default="",
                    help="comma-separated block ids allowed to skip, empty=all")
    ap.add_argument("--max-skips", type=int, default=2)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    adapter = load_adapter_from_source(args.state_path, device, dtype)

    print(f"[calib] collecting w_recent on {args.n_samples} LAMBADA prefixes, seq_len={args.seq_len}")
    per_block = collect_w_recent(base, adapter, tok, device, args.n_samples, args.seq_len)
    thresholds = pick_thresholds(per_block, args.quantile)
    for b in sorted(thresholds):
        vs = per_block[b]
        print(f"  block {b:2d}: τ={thresholds[b]:.4f}  range=[{min(vs):.3f}, {max(vs):.3f}]  n={len(vs)}")

    eligible = None
    if args.eligible:
        eligible = [int(x) for x in args.eligible.split(",")]

    out = {
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "eligible_blocks": eligible,
        "max_skips": args.max_skips,
        "quantile": args.quantile,
        "n_samples": args.n_samples,
        "source": os.path.abspath(args.state_path),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calib] wrote {args.output}")


if __name__ == "__main__":
    main()
