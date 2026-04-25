"""Analyse the 2B_L4 LIBERO VLA ckpt for dynamic-skip feasibility.

Outputs two tables:
  1) Per-block w_recent distribution on real LIBERO samples
     (mean, std, min, q25, q50, q75, q85, q95, max)
  2) Per-block single-skip action drift (MSE between full-forward action
     and action with only block k force-skipped)

The analysis answers:
  - Which blocks have a distribution of w_recent that lets a quantile-based
    threshold separate "safe to skip" vs "must run"?
  - Which blocks, if skipped in isolation, cause minimal action drift?

The intersection of "separable w_recent" and "low drift" defines the
recommended eligible_blocks set. Thresholds come from the per-block q=0.85
quantile; max_skips is chosen so the sum of drifts for the top-k most-safe
blocks stays under a tolerance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/starVLA")
from starVLA.model.framework.base_framework import baseframework  # noqa: E402


def load_libero_samples(
    data_root: Path,
    suite: str = "libero_spatial_no_noops_1.0.0_lerobot",
    n_episodes: int = 16,
    frame_per_episode: int = 2,
):
    """Return list of {"image": PIL.Image, "lang": str} sampled across episodes."""
    import pyarrow.parquet as pq
    import imageio.v3 as iio

    suite_root = data_root / suite
    tasks = {}
    with open(suite_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tasks[d["task_index"]] = d["task"]

    parquets = sorted((suite_root / "data" / "chunk-000").glob("*.parquet"))
    videos = suite_root / "videos" / "chunk-000" / "observation.images.image"

    samples = []
    for pq_path in parquets[:n_episodes]:
        t = pq.read_table(pq_path).to_pandas()
        ep = int(t.iloc[0]["episode_index"])
        task_idx = int(t.iloc[0]["task_index"])
        lang = tasks[task_idx]
        video_path = videos / f"episode_{ep:06d}.mp4"
        if not video_path.exists():
            continue
        frames = iio.imread(video_path, plugin="pyav")  # T,H,W,C
        T = frames.shape[0]
        picks = np.linspace(0, T - 1, frame_per_episode).astype(int)
        for idx in picks:
            img = Image.fromarray(frames[idx])
            samples.append({"image": [img], "lang": lang})
    return samples


def install_recording_router(adapter):
    """Patch router.route to record per-block w_recent each call.
    Returns (records_dict, restore_fn)."""
    records: dict[int, list[float]] = defaultdict(list)
    orig_route = adapter.router.route

    def recording_route(position, completed, _r=records, _orig=orig_route):
        routed, alpha = _orig(position, completed)
        if alpha.shape[-1] >= 2:
            w = float(alpha[..., -1].float().mean().item())
            _r[position - 1].append(w)  # position is 1-indexed block id
        return routed, alpha

    adapter.router.route = recording_route

    def restore():
        adapter.router.route = orig_route

    return records, restore


@torch.no_grad()
def run_no_skip(vla, samples):
    """Forward each sample through predict_action with skip disabled.
    Records w_recent per block and the full-forward action."""
    adapter = vla.attnres_adapter
    records, restore = install_recording_router(adapter)
    actions = []
    try:
        for s in samples:
            out = vla.predict_action(examples=[s], enable_skipping=False)
            actions.append(np.array(out["normalized_actions"]).copy())
    finally:
        restore()
    return records, actions


@torch.no_grad()
def measure_single_block_drift(vla, samples, base_actions, block_idx):
    """Run each sample with a dyn_skip_config that forces block `block_idx`
    to skip (threshold 0) and all other blocks locked (threshold 1e9)."""
    # Force skip on exactly one block by lowering its threshold to -inf
    # while keeping all others at +inf. max_skips=1 means at most this one
    # block skips per forward.
    thr = {b: 1e9 for b in range(vla.attnres_adapter.n_blocks)}
    thr[block_idx] = -1e9  # always trigger
    cfg = {
        "thresholds": thr,
        "eligible_blocks": {block_idx},
        "max_skips": 1,
    }
    mses = []
    for s, base in zip(samples, base_actions):
        out = vla.predict_action(
            examples=[s], enable_skipping=True, dynamic_skip_config=cfg
        )
        a = np.array(out["normalized_actions"])
        mses.append(float(np.mean((a - base) ** 2)))
    return float(np.mean(mses)), float(np.std(mses))


def summarise(records: dict[int, list[float]]):
    """Return per-block distribution stats."""
    out = {}
    for b in sorted(records.keys()):
        vals = np.asarray(records[b], dtype=np.float64)
        out[b] = dict(
            n=len(vals),
            mean=float(vals.mean()),
            std=float(vals.std()),
            min=float(vals.min()),
            q25=float(np.quantile(vals, 0.25)),
            q50=float(np.quantile(vals, 0.50)),
            q75=float(np.quantile(vals, 0.75)),
            q85=float(np.quantile(vals, 0.85)),
            q95=float(np.quantile(vals, 0.95)),
            max=float(vals.max()),
        )
    return out


def print_w_recent_table(stats):
    print("\n== Per-block w_recent distribution (VLA inputs, LIBERO spatial) ==")
    print(
        f"{'blk':>3}  {'n':>4}  {'mean':>6}  {'std':>6}  {'min':>6}  "
        f"{'q25':>6}  {'q50':>6}  {'q75':>6}  {'q85':>6}  {'q95':>6}  {'max':>6}"
    )
    for b, s in stats.items():
        print(
            f"{b:>3}  {s['n']:>4}  {s['mean']:>6.3f}  {s['std']:>6.3f}  "
            f"{s['min']:>6.3f}  {s['q25']:>6.3f}  {s['q50']:>6.3f}  "
            f"{s['q75']:>6.3f}  {s['q85']:>6.3f}  {s['q95']:>6.3f}  {s['max']:>6.3f}"
        )


def print_drift_table(drifts):
    print("\n== Per-block single-skip action drift (MSE vs full-forward) ==")
    print(f"{'blk':>3}  {'mse_mean':>9}  {'mse_std':>9}")
    for b in sorted(drifts.keys()):
        m, s = drifts[b]
        print(f"{b:>3}  {m:>9.3e}  {s:>9.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="/home/user01/Minko/reskip2/reskip/starVLA/results/Checkpoints/libero_pathB_2B_L4_v3_30k/final_model/pytorch_model.pt",
    )
    ap.add_argument(
        "--data-root",
        default="/home/user01/Minko/reskip2/reskip/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA",
    )
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--frame-per-episode", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-drift", action="store_true", help="skip the drift measurement")
    ap.add_argument(
        "--output", default=None, help="optional JSON path to dump all stats"
    )
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    # CUDA_VISIBLE_DEVICES is effective only before torch.cuda init, so we
    # set it via envvar but also use "cuda:0" since VISIBLE_DEVICES remaps.
    device = torch.device("cuda:0")

    print(f"[analyze] loading ckpt {args.ckpt}")
    vla = baseframework.from_pretrained(args.ckpt).to(torch.bfloat16).to(device).eval()

    print(
        f"[analyze] sampling {args.n_episodes} episodes × "
        f"{args.frame_per_episode} frames from libero_spatial"
    )
    samples = load_libero_samples(
        Path(args.data_root),
        n_episodes=args.n_episodes,
        frame_per_episode=args.frame_per_episode,
    )
    print(f"[analyze] collected {len(samples)} (image, instruction) samples")

    print("[analyze] forward pass with router recording (no skip)")
    records, base_actions = run_no_skip(vla, samples)
    w_stats = summarise(records)
    print_w_recent_table(w_stats)

    drifts = {}
    if not args.skip_drift:
        n_blocks = vla.attnres_adapter.n_blocks
        for b in range(n_blocks):
            print(f"[analyze] measuring single-skip drift for block {b}")
            drifts[b] = measure_single_block_drift(vla, samples, base_actions, b)
        print_drift_table(drifts)

    if args.output:
        payload = {
            "ckpt": args.ckpt,
            "n_blocks": vla.attnres_adapter.n_blocks,
            "w_recent": w_stats,
            "drifts": {int(k): {"mse_mean": v[0], "mse_std": v[1]} for k, v in drifts.items()},
            "n_samples": len(samples),
            "suite": "libero_spatial",
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[analyze] dumped stats to {args.output}")


if __name__ == "__main__":
    main()
