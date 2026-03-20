#!/usr/bin/env python3
"""
Export experiment results to LaTeX-includable .tex snippets.

Each snippet contains a single number that can be \\input{} in the paper.
This avoids hardcoding results and auto-updates when experiments are rerun.
"""

import json
import math
import os
import sys


def write_tex(path, value):
    """Write a single value to a .tex file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(value))


def export_standard(output_dir="outputs/standard"):
    """Export standard AttnRes results."""
    results_path = os.path.join(output_dir, "final_results.json")
    if not os.path.exists(results_path):
        print(f"  No results at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Parameters
    n_params = results.get("n_params", 0)
    if n_params > 1e6:
        write_tex(f"{output_dir}/params.tex", f"{n_params/1e6:.1f}M")
    else:
        write_tex(f"{output_dir}/params.tex", f"{n_params:,}")

    # Baseline PPL (eps=0)
    sweep = results.get("skip_sweep", [])
    if sweep:
        baseline = sweep[0]
        write_tex(f"{output_dir}/ppl_baseline.tex", f"{baseline['perplexity']:.2f}")

        # Per-threshold results
        threshold_map = {
            0.01: "eps001", 0.05: "eps005", 0.1: "eps010",
            0.2: "eps020", 0.5: "eps050",
        }
        for entry in sweep:
            eps = entry["threshold"]
            key = threshold_map.get(eps)
            if key:
                ppl = entry["perplexity"]
                flops = entry["flops_ratio"] * 100
                blocks = entry["avg_blocks"]
                delta = ((ppl - baseline["perplexity"]) / baseline["perplexity"]) * 100

                write_tex(f"{output_dir}/ppl_{key}.tex", f"{ppl:.2f}")
                write_tex(f"{output_dir}/flops_{key}.tex", f"{flops:.0f}")
                write_tex(f"{output_dir}/blocks_{key}.tex", f"{blocks:.0f}")
                write_tex(f"{output_dir}/delta_{key}.tex", f"{delta:+.1f}")

    # Best val loss
    best_val = results.get("best_val_loss", 0)
    write_tex(f"{output_dir}/best_val_loss.tex", f"{best_val:.4f}")
    write_tex(f"{output_dir}/best_val_ppl.tex", f"{math.exp(min(best_val, 20)):.2f}")

    # FLOP info
    flops = results.get("flops", {})
    write_tex(f"{output_dir}/gflops.tex", f"{flops.get('total_gflops', 0):.1f}")

    print(f"  Standard: exported {len(sweep)} threshold entries")


def export_looping(output_dir="outputs/looping"):
    """Export looping AttnRes results."""
    results_path = os.path.join(output_dir, "final_results.json")
    if not os.path.exists(results_path):
        print(f"  No results at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Parameters
    total = results.get("total_params", 0)
    unique = results.get("unique_params", 0)
    if total > 1e6:
        write_tex(f"{output_dir}/total_params.tex", f"{total/1e6:.1f}M")
    else:
        write_tex(f"{output_dir}/total_params.tex", f"{total:,}")
    if unique > 1e6:
        write_tex(f"{output_dir}/unique_params.tex", f"{unique/1e6:.1f}M")
    else:
        write_tex(f"{output_dir}/unique_params.tex", f"{unique:,}")

    sharing = (1 - unique / total) * 100 if total > 0 else 0
    write_tex(f"{output_dir}/sharing.tex", f"{sharing:.0f}")

    # PPL
    val_loss = results.get("best_val_loss", 0)
    ppl = results.get("best_val_ppl", math.exp(min(val_loss, 20)))
    write_tex(f"{output_dir}/ppl.tex", f"{ppl:.2f}")

    # Extract effective depth from training log
    log_path = os.path.join(output_dir, "train_log.jsonl")
    if os.path.exists(log_path):
        # Get last epoch entry with depth info
        last_depth = 0
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                if "avg_depth" in entry:
                    last_depth = entry["avg_depth"]
                elif "effective_depth" in entry:
                    last_depth = entry["effective_depth"]
        write_tex(f"{output_dir}/depth.tex", f"{last_depth:.1f}")
    else:
        write_tex(f"{output_dir}/depth.tex", "---")

    print(f"  Looping: exported (PPL={ppl:.2f}, sharing={sharing:.0f}%)")


def export_vla(output_dir="outputs/vla"):
    """Export VLA benchmark results."""
    results_path = os.path.join(output_dir, "vla_results.json")
    if not os.path.exists(results_path):
        print(f"  No results at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Per-mode, per-task results
    mode_map = {"none": "none", "uniform": "uniform", "modality_aware": "modality"}
    for mode_result in results.get("skip_mode_results", []):
        mode = mode_result.get("skip_mode", "")
        key = mode_map.get(mode, mode)

        write_tex(f"{output_dir}/l1_{key}.tex", f"{mode_result['avg_l1_error']:.4f}")

        per_task = mode_result.get("per_task_l1", {})
        write_tex(f"{output_dir}/reach_{key}.tex", f"{per_task.get('reach', 0):.4f}")
        write_tex(f"{output_dir}/push_{key}.tex", f"{per_task.get('push', 0):.4f}")
        write_tex(f"{output_dir}/pp_{key}.tex", f"{per_task.get('pick_place', 0):.4f}")

    # Speedup
    speedup = results.get("speedup", 1.0)
    write_tex(f"{output_dir}/speedup.tex", f"{speedup:.2f}")

    # Best L1
    write_tex(f"{output_dir}/best_l1.tex", f"{results.get('best_val_l1', 0):.4f}")

    print(f"  VLA: exported (speedup={speedup:.2f}x)")


def main():
    print("Exporting experiment results to LaTeX snippets...")
    export_standard()
    export_looping()
    export_vla()
    print("Done.")


if __name__ == "__main__":
    main()
