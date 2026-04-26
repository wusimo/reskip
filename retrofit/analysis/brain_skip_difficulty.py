"""Experiment B (paper §brain-validation): skip-count vs token-difficulty.

Hypothesis (B-H1): on the H_r256_5k retrofit, total skip count per LAMBADA
prefix is *negatively* correlated with mean token NLL (= prediction
difficulty). The transformer analogue of Kar et al. 2019 — easy stimuli
short-circuit, hard stimuli recruit deeper compute.

Outputs:
  --out-csv  : per-prefix (mean_nll, max_token_nll, skip_count)
  --out-fig  : scatter + binned-mean overlay; reports Spearman ρ in caption

Reuses retrofit.eval.eval_dynamic_skip for model loading + threshold
calibration; we run a slimmer single-pass eval that records (skip_count, NLL)
per prefix instead of just aggregating accuracy.

Status: prepared but NOT run. See retrofit/analysis/brain_motivation_design.md
for the run plan.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit")
sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/retrofit/eval")

from eval_dynamic_skip import calibrate_thresholds, load_trained, pick_thresholds


@torch.no_grad()
def per_prefix_skip_and_nll(model, tok, device, eligible, thresholds, max_skips, n=1000):
    """For each LAMBADA prefix, return (mean_nll, max_token_nll, skip_count,
    mean_w_recent_all, mean_w_recent_eligible, per_block_w_recent...).

    The discrete skip_count (capped by max_skips) is a coarse coarsening of
    the underlying continuous router signal. We also report the continuous
    `w_recent` averaged across blocks, both over all blocks and the eligible
    subset, for a finer-grained correlation with NLL.
    """
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if n < len(ds):
        ds = ds.select(range(n))
    cfg = dict(thresholds=thresholds, eligible_blocks=set(eligible), max_skips=max_skips)
    rows = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        text = ex["text"].strip()
        if " " not in text:
            continue
        last = text.rfind(" ")
        ctx = text[:last]
        ctx_ids = tok.encode(ctx, add_special_tokens=False)
        full_ids = tok.encode(text, add_special_tokens=False)
        tgt_ids = full_ids[len(ctx_ids):]
        if not tgt_ids:
            continue
        inp = torch.tensor([ctx_ids + tgt_ids], device=device)
        out = model(input_ids=inp, dynamic_skip_config=cfg)
        logits = out.logits
        start = len(ctx_ids) - 1
        pred_logits = logits[0, start:start + len(tgt_ids), :]
        tgt = torch.tensor(tgt_ids, device=device)
        lp = F.log_softmax(pred_logits.float(), dim=-1)
        token_nll = -lp.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
        skip_count = sum(1 for t in (out.skip_trace or []) if t["skipped"])
        per_block_w = {}
        all_w = []
        elig_w = []
        for t in (out.skip_trace or []):
            b = t.get("block_idx", None)
            w = t.get("w_recent", None)
            if w is None:
                continue
            per_block_w[b] = float(w)
            all_w.append(float(w))
            if b in eligible:
                elig_w.append(float(w))
        row = {
            "idx": i,
            "ctx_len": len(ctx_ids),
            "tgt_len": len(tgt_ids),
            "mean_nll": float(token_nll.mean().item()),
            "max_token_nll": float(token_nll.max().item()),
            "skip_count": skip_count,
            "mean_w_recent_all": (sum(all_w) / len(all_w)) if all_w else float("nan"),
            "mean_w_recent_eligible": (sum(elig_w) / len(elig_w)) if elig_w else float("nan"),
        }
        for b, w in per_block_w.items():
            row[f"w_recent_b{b}"] = w
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ds)}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def spearman_rho(x, y):
    """Pure-python Spearman rank correlation (no scipy dep)."""
    n = len(x)
    if n < 3:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _rank(xs):
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[order[k]] = avg_rank
        i = j + 1
    return rank


def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Union of keys (per-block w_recent_b{n} only present for some blocks)
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_figure(rows, fig_path, rho_mean, rho_max):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    nll = np.array([r["mean_nll"] for r in rows])
    skip = np.array([r["skip_count"] for r in rows])

    # Tercile bins on mean_nll
    q33, q67 = np.quantile(nll, [1.0 / 3, 2.0 / 3])
    bins = np.where(nll <= q33, 0, np.where(nll <= q67, 1, 2))
    bin_means = [skip[bins == k].mean() for k in range(3)]
    bin_sems = [skip[bins == k].std(ddof=1) / np.sqrt(max((bins == k).sum(), 1)) for k in range(3)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    # Scatter
    ax1.scatter(nll, skip + np.random.uniform(-0.12, 0.12, size=len(skip)),
                s=8, alpha=0.35, color="#3b6ea8")
    ax1.set_xlabel("mean token NLL (prediction difficulty)")
    ax1.set_ylabel("dynamic skip count (jittered)")
    ax1.set_title(f"per-prefix: ρ_mean={rho_mean:.3f}  ρ_max={rho_max:.3f}")

    # Binned means
    ax2.bar(["easy\n(NLL≤q33)", "medium", "hard\n(NLL>q67)"],
            bin_means, yerr=bin_sems, capsize=4, color=["#7fbf7b", "#cccccc", "#d6604d"])
    ax2.set_ylabel("mean skip count")
    ax2.set_title("skip count by difficulty tercile")
    for k, m in enumerate(bin_means):
        ax2.text(k, m + 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Difficulty-dependent depth on H_r256_5k retrofit (LAMBADA, $\\mathcal{P}=\\{4,6,11\\}, q=0.85, M=2$)")
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"[brain_skip] figure → {fig_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state-path", required=True)
    p.add_argument("--num-blocks", type=int, default=14)
    p.add_argument("--calib-n", type=int, default=32)
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--quantile", type=float, default=0.85)
    p.add_argument("--eligible", default="4,6,11")
    p.add_argument("--max-skips", type=int, default=2)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-csv", default="outputs/standard/analysis/brain_skip_difficulty.csv")
    p.add_argument("--out-fig", default="paper/figures/brain_skip_difficulty.pdf")
    args = p.parse_args()
    device = f"cuda:{args.gpu}"

    model, tok = load_trained(args.state_path, device, args.num_blocks)
    eligible = [int(x) for x in args.eligible.split(",")]

    print(f"[brain_skip] calibrating thresholds on {args.calib_n} prefixes (q={args.quantile})")
    per_block = calibrate_thresholds(model, tok, device, n_samples=args.calib_n)
    thresholds = pick_thresholds(per_block, q=args.quantile)
    for b in sorted(thresholds):
        print(f"  block {b:2d}: τ={thresholds[b]:.4f}")

    print(f"[brain_skip] running per-prefix eval (n={args.eval_n})")
    rows = per_prefix_skip_and_nll(model, tok, device, eligible, thresholds, args.max_skips, n=args.eval_n)
    write_csv(rows, args.out_csv)
    print(f"[brain_skip] csv → {args.out_csv}  ({len(rows)} rows)")

    nll_mean = [r["mean_nll"] for r in rows]
    nll_max = [r["max_token_nll"] for r in rows]
    skip = [r["skip_count"] for r in rows]
    w_all = [r["mean_w_recent_all"] for r in rows]
    w_elig = [r["mean_w_recent_eligible"] for r in rows]
    rho_mean = spearman_rho(nll_mean, skip)
    rho_max = spearman_rho(nll_max, skip)
    rho_w_all = spearman_rho(nll_mean, w_all)
    rho_w_elig = spearman_rho(nll_mean, w_elig)
    print(f"\n=== RESULT ===")
    print(f"Spearman ρ(mean_nll, skip_count)        = {rho_mean:.4f}  (discrete; capped by M_max)")
    print(f"Spearman ρ(max_token_nll, skip_count)   = {rho_max:.4f}")
    print(f"Spearman ρ(mean_nll, mean_w_recent_all) = {rho_w_all:.4f}  (continuous, all 14 blocks)")
    print(f"Spearman ρ(mean_nll, w_recent_eligible) = {rho_w_elig:.4f}  (continuous, eligible only)")
    print(f"(Hypothesis B-H1 supported if ρ(NLL, w_recent) is positive: harder inputs → router needs more older context → higher overall α on predecessor lower / α more spread / w_recent LOWER → negative ρ. Brain claim direction = ρ < 0.)")

    # Per-block correlations
    print(f"\nPer-block ρ(mean_nll, w_recent_b{{n}}):")
    block_keys = sorted({k for r in rows for k in r.keys() if k.startswith("w_recent_b")},
                       key=lambda s: int(s.replace("w_recent_b", "")))
    for k in block_keys:
        vals = [r.get(k) for r in rows if r.get(k) is not None and not (isinstance(r.get(k), float) and r.get(k) != r.get(k))]
        # skip blocks where some prefixes lack a sample
        if len(vals) < len(rows) * 0.9:
            continue
        idxs = [i for i, r in enumerate(rows) if r.get(k) is not None]
        nlls_ok = [nll_mean[i] for i in idxs]
        rho_b = spearman_rho(nlls_ok, vals)
        print(f"  {k:>14}: ρ={rho_b:+.3f}  (n={len(vals)})")

    make_figure(rows, args.out_fig, rho_mean, rho_max)


if __name__ == "__main__":
    main()
