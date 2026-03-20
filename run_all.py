#!/usr/bin/env python3
"""
Master script: Run all ReSkip experiments end-to-end.

This runs:
1. Standard AttnRes training + skip sweep evaluation
2. Looping AttnRes training + depth analysis
3. VLA benchmark with modality-adaptive depth
4. Routing analysis and visualization

Usage:
    python run_all.py                      # Full pipeline (structured synthetic)
    python run_all.py --dataset wikitext2   # Use WikiText-2
    python run_all.py --quick              # Quick smoke test
    python run_all.py --skip_vla           # Skip VLA benchmark
"""

import argparse
import os
import subprocess
import sys
import time


def run_cmd(cmd, description):
    """Run a command with nice output."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"  CMD: {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {description} ({elapsed:.1f}s)")

    if result.returncode != 0:
        print(f"  WARNING: Command exited with code {result.returncode}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all ReSkip experiments")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test with minimal data")
    parser.add_argument("--dataset", type=str, default="structured_synthetic",
                       choices=["structured_synthetic", "wikitext", "wikitext2"])
    parser.add_argument("--skip_vla", action="store_true", help="Skip VLA benchmark")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip routing analysis")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_base", type=str, default="outputs")
    args = parser.parse_args()

    python = sys.executable
    start_time = time.time()

    # Common args
    if args.quick:
        # Tiny model, few samples, few epochs
        model_args = ["--d_model", "128", "--n_heads", "4", "--n_layers", "4",
                      "--n_blocks", "2", "--seq_len", "64"]
        train_args = ["--epochs", "3", "--batch_size", "16", "--num_train", "2000",
                      "--num_val", "500", "--log_every", "10", "--warmup_steps", "20"]
        vla_model = ["--d_model", "128", "--n_heads", "4", "--n_layers", "4",
                     "--n_blocks", "2", "--seq_len", "16", "--vision_dim", "256",
                     "--vision_seq_len", "16"]
        vla_train = ["--epochs", "3", "--batch_size", "8", "--num_train", "500",
                     "--num_val", "100"]
    else:
        # Default: small but meaningful
        model_args = ["--d_model", "256", "--n_heads", "8", "--n_layers", "8",
                      "--n_blocks", "4", "--seq_len", "256"]
        train_args = ["--epochs", "10", "--batch_size", "32", "--num_train", "50000",
                      "--num_val", "5000", "--log_every", "25", "--warmup_steps", "200"]
        vla_model = ["--d_model", "256", "--n_heads", "8", "--n_layers", "8",
                     "--n_blocks", "4", "--seq_len", "32", "--vision_dim", "512",
                     "--vision_seq_len", "32"]
        vla_train = ["--epochs", "10", "--batch_size", "16", "--num_train", "3000",
                     "--num_val", "500"]

    common = ["--device", args.device, "--dataset", args.dataset]
    results = {}

    # =========================================================================
    # 1. Standard AttnRes Training
    # =========================================================================
    std_dir = os.path.join(args.output_base, "standard")
    success = run_cmd(
        [python, "experiments/train_lm.py", "--mode", "standard",
         "--output_dir", std_dir] + model_args + train_args + common,
        "STEP 1: Train Standard AttnRes Model",
    )
    results["standard_training"] = success

    # =========================================================================
    # 2. Evaluate with skip sweep (if training succeeded)
    # =========================================================================
    if success:
        best_ckpt = os.path.join(std_dir, "best.pt")
        if os.path.exists(best_ckpt):
            run_cmd(
                [python, "experiments/train_lm.py", "--mode", "eval",
                 "--checkpoint", best_ckpt,
                 "--output_dir", std_dir,
                 "--dataset", args.dataset,
                 "--device", args.device] + model_args[:2],  # only need batch/seq for eval
                "STEP 2: Evaluate Standard Model (Skip Sweep)",
            )

    # =========================================================================
    # 3. Looping AttnRes Training
    # =========================================================================
    loop_dir = os.path.join(args.output_base, "looping")
    loop_specific = ["--n_unique_blocks", "2" if args.quick else "4",
                     "--max_loops", "2" if args.quick else "3"]
    success = run_cmd(
        [python, "experiments/train_lm.py", "--mode", "looping",
         "--output_dir", loop_dir] + model_args + train_args + common + loop_specific,
        "STEP 3: Train Looping AttnRes Model (ReLoop)",
    )
    results["looping_training"] = success

    # =========================================================================
    # 4. VLA Benchmark
    # =========================================================================
    if not args.skip_vla:
        vla_dir = os.path.join(args.output_base, "vla")
        success = run_cmd(
            [python, "experiments/benchmark_vla.py",
             "--output_dir", vla_dir,
             "--device", args.device] + vla_model + vla_train,
            "STEP 4: VLA Benchmark (Modality-Adaptive Depth)",
        )
        results["vla_benchmark"] = success

    # =========================================================================
    # 5. Routing Analysis & Visualization
    # =========================================================================
    if not args.skip_analysis:
        best_ckpt = os.path.join(std_dir, "best.pt")
        results_json = os.path.join(std_dir, "final_results.json")

        if os.path.exists(best_ckpt):
            analysis_args = [
                python, "experiments/analyze_routing.py",
                "--checkpoint", best_ckpt,
                "--device", args.device,
            ]
            if os.path.exists(results_json):
                analysis_args += ["--sweep_results", results_json]

            success = run_cmd(
                analysis_args,
                "STEP 5: Routing Analysis & Visualization",
            )
            results["routing_analysis"] = success

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE ({total_time:.1f}s total)")
    print(f"{'='*70}")

    for step, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {step}")

    print(f"\nOutputs:")
    for d in [std_dir, loop_dir]:
        if os.path.exists(d):
            files = os.listdir(d)
            print(f"  {d}/: {len(files)} files")
            for f in sorted(files):
                size = os.path.getsize(os.path.join(d, f))
                if size > 1024*1024:
                    print(f"    {f} ({size/1024/1024:.1f}MB)")
                elif size > 1024:
                    print(f"    {f} ({size/1024:.0f}KB)")
                else:
                    print(f"    {f} ({size}B)")

    if not args.skip_vla:
        vla_dir = os.path.join(args.output_base, "vla")
        if os.path.exists(vla_dir):
            files = os.listdir(vla_dir)
            print(f"  {vla_dir}/: {len(files)} files")

    analysis_dir = os.path.join(std_dir, "analysis")
    if os.path.exists(analysis_dir):
        print(f"  {analysis_dir}/:")
        for f in sorted(os.listdir(analysis_dir)):
            print(f"    {f}")


if __name__ == "__main__":
    main()
