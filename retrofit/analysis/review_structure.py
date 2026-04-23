"""Structural review of trained retrofit: what does data actually flow through?

For each block n, compute:
  gamma_n                    — gate value
  ||r_n - h_{n-1}|| / ||h||  — size of raw AttnRes routing "correction signal"
  ||x_n - h_{n-1}|| / ||h||  — actual delta added (gamma * adapter(r - h))
  alpha concentration        — where does the router attend? (embed, recent, etc.)

And overall: logits divergence retrofit vs base.
"""
import sys
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
import argparse
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


MODEL = "/home/user01/Minko/models/Qwen3-VL-2B"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-path", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    device = f"cuda:{args.gpu}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Separate un-patched model for true base comparison
    base_clean = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    # Wrapped model for retrofit
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    ck = torch.load(args.state_path, map_location="cpu")
    cfg = ck.get("config", {})
    kwargs = {"num_blocks": cfg.get("num_blocks", 14)}
    if "adapter_rank" in cfg:
        kwargs["adapter_rank"] = cfg["adapter_rank"]
    no_adapter = cfg.get("no_adapter", not ck["adapters"])
    if no_adapter:
        kwargs["no_adapter"] = True
        kwargs.pop("adapter_rank", None)
    model = Qwen3VLAttnResRetrofit(base, **kwargs).to(device, dtype=torch.bfloat16)
    model.router.load_state_dict({k: v.to(device, dtype=torch.bfloat16) for k, v in ck["router"].items()})
    if not no_adapter:
        model.adapters.load_state_dict({k: v.to(device, dtype=torch.bfloat16) for k, v in ck["adapters"].items()})
    model.gamma.data.copy_(ck["gamma"].to(device, dtype=torch.bfloat16))
    model.eval()

    gammas = model.gamma.detach().float().cpu().tolist()
    print("=== γ per block ===")
    for i, g in enumerate(gammas):
        print(f"  block {i:2d}: γ = {g:+.4f}")

    text = "The quick brown fox jumps over the lazy dog. The capital of France is"
    ids = tok(text, return_tensors="pt").input_ids.to(device)

    rf_block_inputs = {}
    rf_routed = {}
    rf_alphas = {}
    orig = model._compute_block_input
    def patched(bidx, prev, completed, collect_alpha=True):
        corrected, alpha, routed, ent = orig(bidx, prev, completed, collect_alpha=True)
        rf_block_inputs[bidx] = (
            corrected.float().clone() if corrected is not None else None,
            prev.float().clone(),
        )
        if routed is not None:
            rf_routed[bidx] = routed.float().clone()
        if alpha is not None:
            rf_alphas[bidx] = alpha.float().clone()
        return corrected, alpha, routed, ent
    model._compute_block_input = patched

    with torch.no_grad():
        out = model(input_ids=ids, return_alpha=True)
    model._compute_block_input = orig

    print()
    print("=== Per-block data flow ===")
    header = "blk    gamma  | r-h |/|h|  | Δx |/|h|  α[embed]  α[recent]  α top-src"
    print(header)
    print("-" * len(header))
    for n in range(14):
        if n not in rf_block_inputs:
            continue
        corrected, prev = rf_block_inputs[n]
        if corrected is None:
            continue
        dx = corrected - prev
        rel_dx = (dx.norm() / (prev.norm() + 1e-8)).item()
        if n in rf_routed:
            delta = rf_routed[n] - prev
            rel_delta = (delta.norm() / (prev.norm() + 1e-8)).item()
        else:
            rel_delta = 0.0
        a = rf_alphas.get(n)
        if a is not None and a.shape[-1] >= 2:
            a_mean = a.mean(dim=(0, 1))
            a_embed = float(a_mean[0].item())
            a_recent = float(a_mean[-1].item())
            top_src = int(a_mean.argmax().item())
            top_label = "embed" if top_src == 0 else f"b{top_src-1}"
        else:
            a_embed = a_recent = 0.0
            top_label = "-"
        print(f"{n:>3}  {gammas[n]:+7.4f}  {rel_delta:>11.4f}  {rel_dx:>10.6f}  {a_embed:>8.4f}  {a_recent:>9.4f}  {top_label}")

    with torch.no_grad():
        base_logits = base_clean(input_ids=ids, use_cache=False).logits
    r_logits = out.logits
    rel = ((base_logits.float() - r_logits.float()).norm() / base_logits.float().norm()).item()
    argmatch = (base_logits.argmax(-1) == r_logits.argmax(-1)).all().item()
    print()
    print(f"Overall logits rel diff: {rel:.6f}   argmax match: {argmatch}")
    print()
    mean_abs_g = sum(abs(x) for x in gammas) / len(gammas)
    print(f"γ stats: range=[{min(gammas):+.4f}, {max(gammas):+.4f}]  mean|γ|={mean_abs_g:.4f}")
    # Effective AttnRes participation: avg(|Δx|) / avg(|h|) → fraction of block input that is AttnRes-derived
    rel_dxs = []
    for n in rf_block_inputs:
        corrected, prev = rf_block_inputs[n]
        if corrected is None: continue
        rel_dxs.append((corrected - prev).norm().item() / (prev.norm().item() + 1e-8))
    if rel_dxs:
        print(f"Effective AttnRes participation (avg |Δx|/|h|): {sum(rel_dxs)/len(rel_dxs):.6f}")
        print(f"  Interpretation: {sum(rel_dxs)/len(rel_dxs)*100:.3f}% of block-input magnitude comes from AttnRes correction;")
        print(f"  the remainder (~{100 - sum(rel_dxs)/len(rel_dxs)*100:.2f}%) is the original h_{{n-1}} residual path.")


if __name__ == "__main__":
    main()
