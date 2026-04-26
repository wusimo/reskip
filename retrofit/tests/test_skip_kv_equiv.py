"""Pin down: does our skip K/V-only path write cache values identical to a
full-layer forward that wrote its natural K/V?

Both runs are SINGLE forward passes (no iterative decode). The only difference
is use_cache flag. If our skip path is correct, the last-position logits under
use_cache=True must equal the use_cache=False logits (both do the same compute;
cache just records K/V as a side effect).

If logits differ meaningfully, the K/V we inject at skipped layers is different
from what the layer would have produced, and multi-step decode will drift.
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit

MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"


def load(state_path, device):
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    ck = torch.load(state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kw = dict(num_blocks=cfg.get("num_blocks", 14))
    if "adapter_rank" in cfg:
        kw["adapter_rank"] = cfg["adapter_rank"]
    model = Qwen3VLAttnResRetrofit(base, **kw).to(device=device, dtype=dtype).eval()
    model.router.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["router"].items()})
    model.adapters.load_state_dict({k: v.to(device=device, dtype=dtype) for k, v in ck["adapters"].items()})
    model.gamma.data.copy_(ck["gamma"].to(device=device, dtype=dtype))
    return model, tok


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-path", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    device = f"cuda:{args.gpu}"

    model, tok = load(args.state_path, device)
    ids = tok("The capital of France is Paris. The capital of Germany is",
              return_tensors="pt").input_ids.to(device)

    for cfg in [[4], [4, 10], [4, 10, 12], [2, 4, 10]]:
        model._active_skip_blocks = set(cfg)
        model._dynamic_skip_config = None
        model._collect_block_states = False
        out_nc = model.base_model(input_ids=ids, use_cache=False)
        out_c = model.base_model(input_ids=ids, use_cache=True)
        # Compare last-position logits (used for next-token prediction).
        last_nc = out_nc.logits[:, -1, :].float()
        last_c = out_c.logits[:, -1, :].float()
        diff = (last_nc - last_c).abs().max().item()
        argmax_match = bool((last_nc.argmax(-1) == last_c.argmax(-1)).all().item())
        print(f"skip={cfg}: last-pos logits max|Δ|={diff:.6f}  argmax_match={argmax_match}")


if __name__ == "__main__":
    main()
