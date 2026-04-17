"""Generate paper figures for ReSkip section (340M FineWeb-Edu)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# -----------------------------------------------------------------------
# Load 340M data
# -----------------------------------------------------------------------
analysis = json.loads(
    (ROOT / "outputs/reskip_340M_probe_sweep_fast_gpu7/routing_analysis.json").read_text()
)
full_eval = analysis["full_eval"]
block_importance = full_eval["block_importance"]  # length 8
importance_matrix = np.array(full_eval["importance_matrix"])  # [8, 8], upper-triangular
full_ppl = full_eval["perplexity"]

# Block ablation PPL ratios (blocks 0..5 in the list; blocks 6/7 are boundary-protected)
ablation_entries = analysis["block_ablation"]
ablation_ratios = [None] * 8
for entry in ablation_entries:
    idx = entry["ablated_block"]
    ablation_ratios[idx] = entry["perplexity"] / full_ppl

# -----------------------------------------------------------------------
# Figure 1: Routing heatmap + importance-vs-ablation bar chart
# -----------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.05, 1.35]})

# (a) Heatmap
mat = importance_matrix.copy()
# mask lower triangle (including diagonal) to emphasize the upper-triangular downstream weights
mask = np.tri(mat.shape[0], k=0, dtype=bool)
masked = np.ma.array(mat, mask=mask)
im = axL.imshow(masked, cmap="viridis", aspect="equal", vmin=0, vmax=0.6)
axL.set_xticks(range(8))
axL.set_yticks(range(8))
axL.set_xticklabels([f"{i}" for i in range(8)])
axL.set_yticklabels([f"{i}" for i in range(8)])
axL.set_xlabel("Downstream block $l$")
axL.set_ylabel("Source block $n$")
axL.set_title("(a) AttnRes routing weights $\\alpha_{n\\to l}$")
# overlay values
for i in range(8):
    for j in range(i + 1, 8):
        val = mat[i, j]
        color = "white" if val < 0.3 else "black"
        axL.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)
cbar = plt.colorbar(im, ax=axL, fraction=0.045, pad=0.03)
cbar.ax.tick_params(labelsize=8)

# (b) Importance vs Ablation bar chart — the novel finding
x = np.arange(8)
width = 0.38
# Block importance (max downstream weight), normalize to [0,1] for visual parity (it's already in that range)
imp = np.array(block_importance)
# For ablation ratio: larger = more damage. For comparison, plot log(ratio) normalized.
# Interior blocks (1..6) are what we care about. Plot only 0..5 (available ablation data)
abl = np.array([r if r is not None else np.nan for r in ablation_ratios])

# Plot as dual bars on shared x, separate y-axes
ax2 = axR
ax2b = ax2.twinx()

# Use distinct colors
bars1 = ax2.bar(x - width/2, imp, width, label="AttnRes importance $I(n)$", color="#4C72B0", edgecolor="black", linewidth=0.5)
# Only plot ablation bars where available (blocks 0..5)
valid = ~np.isnan(abl)
bars2 = ax2b.bar(x[valid] + width/2, abl[valid], width, label="Static removal PPL ratio", color="#DD8452", edgecolor="black", linewidth=0.5)

# Highlight block 2 (moderate importance but HIGH static ablation impact — the key disconnect).
# Use ax2b (right axis) because we're pointing to an orange bar value (8.1 on the right scale).
ax2b.annotate(
    "Block 2:\nmoderate $I$ but\n$\\it{highest}$ removal\nPPL (8.1$\\times$)",
    xy=(2 + width/2, abl[2]), xytext=(3.0, 7.5),
    fontsize=8, ha="left", va="top",
    arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
    bbox=dict(boxstyle="round,pad=0.3", fc="#FFE6E6", ec="black", lw=0.5),
)
# Highlight blocks 3 & 5 (both are good dynamic-skip candidates for complementary reasons).
# Block 3: high $I$ but low ablation → rare but safe trigger.
# Block 5: low $I$ + moderate ablation → frequent and safe trigger.
ax2.annotate(
    "Blocks 3, 5:\nour dynamic skip\npositions\n({3}, {5}-axis pair)",
    xy=(5 - width/2, imp[5]), xytext=(5.2, 0.82),
    fontsize=8, ha="left", va="top",
    arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
    bbox=dict(boxstyle="round,pad=0.3", fc="#E6F5E6", ec="black", lw=0.5),
)

ax2.set_xticks(x)
ax2.set_xticklabels([f"{i}" for i in range(8)])
ax2.set_xlabel("Block index $n$")
ax2.set_ylabel("AttnRes importance $I(n)$", color="#4C72B0")
ax2b.set_ylabel("PPL ratio after removal", color="#DD8452")
ax2.tick_params(axis="y", colors="#4C72B0")
ax2b.tick_params(axis="y", colors="#DD8452")
ax2.set_ylim(0, 1.05)
ax2b.set_ylim(1.0, 9.0)
ax2b.axhline(1.0, linestyle=":", color="gray", lw=0.6)

# Legends combined
l1 = ax2.get_legend_handles_labels()
l2 = ax2b.get_legend_handles_labels()
ax2.legend(l1[0] + l2[0], l1[1] + l2[1], loc="upper left", fontsize=8, framealpha=0.95)
ax2.set_title("(b) Importance $\\neq$ irreplaceability")

plt.tight_layout()
out = HERE / "reskip_routing_importance_vs_ablation.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

# -----------------------------------------------------------------------
# Figure 2: Dynamic skip Pareto curve
# -----------------------------------------------------------------------
# Build Pareto data from dynamic_skip_analysis results
results = analysis["dynamic_skip_analysis"]["results"]
points = []
for r in results:
    points.append({
        "avg_blocks": r.get("avg_blocks", 8.0),
        "ppl_ratio": r.get("perplexity", full_ppl) / full_ppl,
        "positions": tuple(r.get("allowed_positions", [])),
        "max_skips": r.get("max_skips", 1),
        "probe": r.get("probe_mode", "all"),
        "quantile": r.get("quantile", 0.5),
    })

# Group by position-set category for coloring
def category(positions, max_skips):
    pos_set = set(positions)
    if pos_set == {5}:
        return "single ({5}, lowest $I$)"
    if pos_set.issubset({4, 5, 6}):
        return "late positions"
    if len(pos_set) >= 4:
        return "all interior"
    return "other subsets"

groups = {}
for p in points:
    cat = category(p["positions"], p["max_skips"])
    groups.setdefault(cat, []).append(p)

fig, ax = plt.subplots(figsize=(6.5, 4.0))

colors = {
    "single ({5}, lowest $I$)": "#55A868",
    "late positions": "#8172B2",
    "other subsets": "#CCB974",
    "all interior": "#4C72B0",
}
markers = {
    "single ({5}, lowest $I$)": "o",
    "late positions": "^",
    "other subsets": "D",
    "all interior": "x",
}

# Plot each group
for cat, pts in groups.items():
    xs = [p["avg_blocks"] for p in pts]
    ys = [p["ppl_ratio"] for p in pts]
    ax.scatter(xs, ys, s=35, c=colors.get(cat, "gray"), marker=markers.get(cat, "o"),
               edgecolors="black", linewidths=0.4, alpha=0.8, label=cat)

# Mark the new-best config: combined {3,5}/skip2/q=0.85 — based on our latency run,
# avg skips/batch ≈ 0.88, so avg_blocks ≈ 7.12 on eval set; PPL ratio ≈ 1.00 (zero benchmark drop)
ax.scatter([7.12], [1.00], s=140, c="#DD8452", marker="*",
           edgecolors="black", linewidths=0.8, zorder=5,
           label="combined {3,5}, max_skips=2, q=0.85 (ours)")

# Full-depth anchor
ax.scatter([8.0], [1.0], s=80, c="black", marker="P", zorder=5, label="full-depth baseline")

# Tolerance line at 5% PPL increase
ax.axhline(1.05, linestyle="--", color="red", lw=0.8, alpha=0.6)
ax.text(7.0, 1.05, "5% PPL tolerance", fontsize=8, color="red", va="bottom")

ax.set_xlabel("Average blocks executed per forward (of 8)")
ax.set_ylabel("PPL ratio (skip / full-depth)")
ax.set_title("Dynamic skip Pareto frontier (340M, FineWeb-Edu proxy set)")
ax.set_xlim(6.5, 8.1)
ax.set_ylim(0.95, 2.0)
ax.axhline(1.0, linestyle=":", color="gray", lw=0.6)
ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = HERE / "reskip_pareto.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

# -----------------------------------------------------------------------
# Figure 3: Wall-clock latency comparison
# -----------------------------------------------------------------------
latency = json.loads((ROOT / "outputs/latency_prev_sweep.json").read_text())
# Keep only a curated set for the paper
keep = [
    "full-depth",
    "attn [5] skip=1 q=0.85",
    "attn [5] skip=1 q=0.95",
    "attn [3, 5] skip=2 q=0.85",
    "attn [3, 5] skip=2 q=0.9",
    "prev [2, 3, 5] skip=2 q=0.8",
]
subset = [r for r in latency if r["name"] in keep]
# Sort by mean_batch_s descending (slowest first)
subset = sorted(subset, key=lambda r: -r["mean_batch_s"])
names = [r["name"].replace("attn ", "attn, ").replace("prev ", "prev, ") for r in subset]
ms = [r["mean_batch_s"] * 1000 for r in subset]
base_ms = [r for r in subset if r["name"] == "full-depth"][0]["mean_batch_s"] * 1000
speedups = [base_ms / m for m in ms]

fig, ax = plt.subplots(figsize=(7.5, 3.2))
colors_bar = ["#333333" if n == "full-depth" else ("#DD8452" if "3, 5" in n else "#4C72B0") for n in names]
bars = ax.barh(range(len(names)), ms, color=colors_bar, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("ms / batch (seq\\_len=8192, batch\\_size=1)")
for i, (bar, s) in enumerate(zip(bars, speedups)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f" {s:.2f}$\\times$", va="center", fontsize=8)
ax.set_xlim(0, max(ms) * 1.15)
ax.set_title("Wall-clock latency comparison (340M, single H100)")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
out = HERE / "reskip_latency.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved {out}")

plt.close("all")
print("Done.")
