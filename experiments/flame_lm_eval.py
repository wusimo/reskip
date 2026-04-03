from __future__ import annotations

import argparse
import runpy
import sys

from flame_reskip_common import FLA_ROOT, FLAME_ROOT  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm_eval with custom ReSkip flame models.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    model_args = f"pretrained={args.model_path}"
    argv = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        args.tasks,
        "--batch_size",
        args.batch_size,
        "--device",
        args.device,
    ]
    if args.output_path:
        argv.extend(["--output_path", args.output_path])
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    if args.num_fewshot is not None:
        argv.extend(["--num_fewshot", str(args.num_fewshot)])
    if args.log_samples:
        argv.append("--log_samples")
    if args.extra:
        argv.extend(args.extra)

    sys.argv = argv
    runpy.run_module("lm_eval", run_name="__main__")


if __name__ == "__main__":
    main()
