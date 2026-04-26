"""Post-process a LIBERO-sim routing dump (JSONL from eval_libero.py
--args.routing-dump-path) into a dyn_skip_config JSON whose thresholds are
per-block quantiles of the ACTUAL sim-input w_recent distribution.

Workflow:
  1. Launch a policy server whose code includes w_recents in the summary.
     (starVLA/src/starvla_integration.py — already patched.)
  2. Run eval_libero.py with --args.routing-dump-path=sim_dump.jsonl on a
     short calibration slice (few trials × few tasks, all 4 suites).
     enable_skipping should be False for calibration (otherwise the dump
     reflects the already-skipping distribution, biasing downstream
     thresholds).
  3. Run this script on the dump to produce a dyn_skip_config JSON.
  4. Use that JSON in a fresh reskip eval.

The per-block stats are computed across the UNION of all forwards in the
dump — suites and tasks pooled — because the final reskip eval runs on the
same 4 suites and a pooled quantile is what matches that mixture.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="JSONL from eval_libero.py --args.routing-dump-path")
    ap.add_argument("--output", required=True, help="dyn_skip_config JSON path")
    ap.add_argument("--quantile", type=float, default=0.85, help="τ_n = q-quantile of w_recent(n)")
    ap.add_argument("--eligible", default="1,4",
                    help="comma-separated block indices P (eligible_blocks)")
    ap.add_argument("--max-skips", type=int, default=2, help="M (cap per forward)")
    ap.add_argument("--notes", default="", help="free-text note embedded in JSON")
    args = ap.parse_args()

    per_block: dict[int, list[float]] = {}
    n_forwards = 0
    with open(args.dump) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ws = d.get("w_recents") or []
            for b, w in enumerate(ws):
                if w is None or (isinstance(w, float) and w != w):
                    continue
                per_block.setdefault(b, []).append(float(w))
            n_forwards += 1

    if not per_block:
        raise SystemExit("no w_recents found in dump")

    print(f"[sim-calib] {n_forwards} forwards, {len(per_block)} blocks")
    print(f"{'blk':>3}  {'n':>5}  {'mean':>6}  {'std':>6}  {'q50':>6}  {'q'+str(int(args.quantile*100)):>6}")
    thresholds = {}
    for b in sorted(per_block):
        v = np.asarray(per_block[b])
        tau = float(np.quantile(v, args.quantile))
        thresholds[str(b)] = tau
        print(f"{b:>3}  {len(v):>5}  {v.mean():>6.3f}  {v.std():>6.3f}  "
              f"{np.quantile(v,0.5):>6.3f}  {tau:>6.3f}")

    eligible = [int(x) for x in args.eligible.split(",") if x.strip()]
    out = {
        "thresholds": thresholds,
        "eligible_blocks": eligible,
        "max_skips": args.max_skips,
        "strategy": "recent_weight_gt",
        "quantile": args.quantile,
        "notes": args.notes or "sim-trajectory calibration (retrofit Part-1 protocol)",
        "source_dump": str(Path(args.dump).resolve()),
        "n_forwards": n_forwards,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[sim-calib] wrote {args.output}")


if __name__ == "__main__":
    main()
