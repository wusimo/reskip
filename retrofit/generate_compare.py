"""Qualitative generation comparison: base vs α-pruned Qwen3-VL-2B."""
import sys, torch
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
from transformers import AutoModelForImageTextToText, AutoTokenizer
from prune_qwen3vl import PrunedQwen3VL

MODEL = "/home/user01/Minko/models/Qwen3-VL-2B"
tok = AutoTokenizer.from_pretrained(MODEL)


def make_model(skip_layers=None):
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    if not skip_layers:
        return base
    return PrunedQwen3VL(base, skip_layers=set(skip_layers)).to("cuda", dtype=torch.bfloat16).eval()


@torch.no_grad()
def generate(model, prompt, max_new=80, temperature=0.7):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = ids
    for _ in range(max_new):
        # Custom forward: for PrunedQwen3VL we use the forward method
        if isinstance(model, PrunedQwen3VL):
            res = model(input_ids=out, labels=None)
            logits = res["logits"][:, -1, :]
        else:
            res = model(input_ids=out, use_cache=False)
            logits = res.logits[:, -1, :]
        logits = logits / temperature
        probs = torch.softmax(logits.float(), dim=-1)
        next_tok = torch.multinomial(probs, 1)
        out = torch.cat([out, next_tok], dim=1)
        if next_tok.item() == tok.eos_token_id:
            break
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


prompts = [
    "The capital city of France is",
    "In Python, to sort a list in reverse order, you can use",
    "Photosynthesis is the process by which plants",
    "The key difference between supervised and unsupervised learning is",
]


def main():
    import argparse
    p_ = argparse.ArgumentParser()
    p_.add_argument("--skip-layers", default="8,9,10,11,12,13,14,15")
    p_.add_argument("--label", default="PRUNED")
    args = p_.parse_args()
    skip = [int(x) for x in args.skip_layers.split(",")]

    print("Loading BASE model...")
    base = make_model(skip_layers=None)
    print(f"Loading {args.label} (skip layers {skip})...")
    pruned = make_model(skip_layers=skip)

    torch.manual_seed(42)

    for p in prompts:
        print(f"\n{'='*70}")
        print(f"PROMPT: {p}")
        print("-" * 70)
        torch.manual_seed(42)
        print(f"BASE:   {generate(base, p)}")
        torch.manual_seed(42)
        print(f"{args.label}: {generate(pruned, p)}")


if __name__ == "__main__":
    main()
