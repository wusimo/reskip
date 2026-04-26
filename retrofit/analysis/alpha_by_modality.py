"""Experiment C (paper §brain-validation, supporting): cross-modality routing
distribution.

Hypothesis (C-H1): per-block α distribution differs across modalities
(text / VL / action) on a single retrofit. The router has implicitly developed
modality-specific pathways — analogous to dorsal/ventral cortical
specialisation [buschman2007topdown, bastos2012canonical].

We have indirect evidence already (LAMBADA-calibrated thresholds tank LIBERO
to 64%); this script makes it direct and visual.

Outputs:
  --out-npz : per-modality per-block w_recent samples (raw)
  --out-fig : per-block stacked histogram + per-block KL divergence

Status: prepared but NOT run. See retrofit/analysis/brain_motivation_design.md
for the run plan.

Implementation notes
--------------------
- Text modality (LAMBADA): load H_r256_5k via load_trained() from
  eval_dynamic_skip; install recording router on the VLM retrofit; run
  forward on LAMBADA prefixes (no skip enabled).
- VL modality (MMStar): same VLM retrofit; pass image+question via the
  Qwen3-VL processor to get multimodal inputs.
- Action modality (LIBERO Spatial): load the VLA H_4B_r256_5k (or a 2B VLA
  ckpt if available) via baseframework; reuse install_recording_router
  pattern from analyze_vla_reskip_2b_l4.py; run predict_action on a small
  number of LIBERO frames.

The recording router copies α[..., -1].mean() into a per-block list; we
flatten across batch+positions, dedup nothing, and dump the raw arrays.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit/eval")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/starVLA")

from datasets import load_dataset

from eval_dynamic_skip import load_trained as load_vlm_retrofit


# Reuse the recording-router pattern from analyze_vla_reskip_2b_l4.py
def install_recording_router(adapter):
    """Patch router.route to record per-block α[..., -1].mean() each call.
    Returns (records_dict, restore_fn)."""
    records: dict[int, list[float]] = defaultdict(list)
    orig_route = adapter.router.route

    def recording_route(position, completed, _r=records, _orig=orig_route):
        routed, alpha = _orig(position, completed)
        if alpha.shape[-1] >= 2:
            w = float(alpha[..., -1].float().mean().item())
            _r[position - 1].append(w)
        return routed, alpha

    adapter.router.route = recording_route

    def restore():
        adapter.router.route = orig_route

    return records, restore


@torch.no_grad()
def collect_text_alpha(model, tok, device, n=200, seq_len=512):
    """LAMBADA text prefixes through VLM retrofit (no images)."""
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test").select(range(n))
    records, restore = install_recording_router(model)
    try:
        for ex in ds:
            text = ex["text"].strip()
            ids = tok.encode(text, add_special_tokens=False)[:seq_len]
            inp = torch.tensor([ids], device=device)
            _ = model(input_ids=inp)
    finally:
        restore()
    return {b: list(v) for b, v in records.items()}


@torch.no_grad()
def collect_vl_alpha(model, tok, device, n=200):
    """MMStar (image + question) through VLM retrofit. Uses the Qwen3-VL
    processor for multimodal tokenisation."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("/home/user01/Minko/models/Qwen3-VL-2B")
    ds = load_dataset("Lin-Chen/MMStar", split="val").select(range(n))
    records, restore = install_recording_router(model)
    try:
        for ex in ds:
            img: Image.Image = ex["image"]
            q = ex["question"]
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": q},
                ]},
            ]
            chat = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            batch = proc(text=[chat], images=[img], return_tensors="pt").to(device)
            _ = model(**batch)
    finally:
        restore()
    return {b: list(v) for b, v in records.items()}


@torch.no_grad()
def collect_action_alpha(vla_ckpt, data_root, n_episodes=8, frame_per_episode=4, device="cuda:0"):
    """LIBERO Spatial frames through VLA retrofit (action modality)."""
    # Reuse load_libero_samples + the predict_action pipeline from analyze_vla_reskip_2b_l4.
    from analyze_vla_reskip_2b_l4 import load_libero_samples
    from starVLA.model.framework.base_framework import baseframework

    samples = load_libero_samples(
        Path(data_root), suite="libero_spatial_no_noops_1.0.0_lerobot",
        n_episodes=n_episodes, frame_per_episode=frame_per_episode,
    )
    vla = baseframework.from_pretrained(vla_ckpt).to(device).eval()
    records, restore = install_recording_router(vla.attnres_adapter)
    try:
        for s in samples:
            _ = vla.predict_action(examples=[s], enable_skipping=False)
    finally:
        restore()
    return {b: list(v) for b, v in records.items()}


def kl_divergence(p_samples, q_samples, n_bins=30):
    """Histogram-based KL(p || q) on [0, 1] support; small additive smoothing."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    p, _ = np.histogram(p_samples, bins=edges, density=False)
    q, _ = np.histogram(q_samples, bins=edges, density=False)
    p = (p + 1.0) / (p.sum() + n_bins)
    q = (q + 1.0) / (q.sum() + n_bins)
    return float(np.sum(p * np.log(p / q)))


def make_figure(by_mod, fig_path):
    """Per-block stacked histogram of w_recent across modalities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blocks = sorted({b for d in by_mod.values() for b in d.keys()})
    n_blocks = len(blocks)
    cols = 4
    rows = (n_blocks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.0 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = {"text": "#3b6ea8", "vl": "#7fbf7b", "action": "#d6604d"}

    for ax_i, b in enumerate(blocks):
        ax = axes[ax_i]
        for mod, color in colors.items():
            xs = by_mod.get(mod, {}).get(b, [])
            if not xs:
                continue
            ax.hist(xs, bins=np.linspace(0, 1, 31), alpha=0.5, color=color, label=mod, density=True)
        ax.set_title(f"block {b}")
        if ax_i == 0:
            ax.legend(fontsize=7)
    for k in range(n_blocks, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Per-block $w_{\\text{recent}}$ distribution by modality (H_r256_5k retrofit)")
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"[alpha_by_modality] figure → {fig_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vlm-state-path", required=True,
                   help="H_r256_5k retrofit state.pt (used for text+VL)")
    p.add_argument("--vla-state-path", default=None,
                   help="VLA retrofit ckpt (used for action). If None, skip action.")
    p.add_argument("--num-blocks", type=int, default=14)
    p.add_argument("--lambada-n", type=int, default=200)
    p.add_argument("--mmstar-n", type=int, default=200)
    p.add_argument("--libero-n-episodes", type=int, default=8)
    p.add_argument("--libero-frame-per-ep", type=int, default=4)
    p.add_argument("--libero-data-root",
                   default="/home/user01/Minko/reskip2/reskip/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-npz", default="outputs/standard/analysis/alpha_by_modality.npz")
    p.add_argument("--out-fig", default="paper/figures/alpha_by_modality.pdf")
    args = p.parse_args()
    device = f"cuda:{args.gpu}"

    by_mod = {}

    # Text + VL: same VLM retrofit
    print(f"[alpha_by_modality] loading VLM retrofit ← {args.vlm_state_path}")
    model, tok = load_vlm_retrofit(args.vlm_state_path, device, args.num_blocks)

    print(f"[alpha_by_modality] text modality: LAMBADA n={args.lambada_n}")
    by_mod["text"] = collect_text_alpha(model, tok, device, n=args.lambada_n)

    print(f"[alpha_by_modality] VL modality: MMStar n={args.mmstar_n}")
    by_mod["vl"] = collect_vl_alpha(model, tok, device, n=args.mmstar_n)

    if args.vla_state_path:
        print(f"[alpha_by_modality] action modality: LIBERO Spatial "
              f"({args.libero_n_episodes} eps × {args.libero_frame_per_ep} frames)")
        by_mod["action"] = collect_action_alpha(
            args.vla_state_path, args.libero_data_root,
            n_episodes=args.libero_n_episodes,
            frame_per_episode=args.libero_frame_per_ep,
            device=device,
        )
    else:
        print("[alpha_by_modality] no --vla-state-path; skipping action modality")

    # Persist raw samples
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **{
        f"{mod}_block{b}": np.asarray(vs)
        for mod, recs in by_mod.items() for b, vs in recs.items()
    })
    print(f"[alpha_by_modality] raw → {args.out_npz}")

    # KL summary
    print("\n=== Per-block KL divergence (text vs other modalities) ===")
    blocks = sorted({b for d in by_mod.values() for b in d.keys()})
    for b in blocks:
        text = by_mod.get("text", {}).get(b, [])
        if not text:
            continue
        line = f"  block {b:2d}: "
        for mod in ("vl", "action"):
            xs = by_mod.get(mod, {}).get(b, [])
            if xs:
                kl = kl_divergence(text, xs)
                line += f"KL(text||{mod})={kl:.3f}  "
        print(line)

    make_figure(by_mod, args.out_fig)


if __name__ == "__main__":
    main()
