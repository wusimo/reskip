from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from flame_reskip_common import FLA_ROOT, FLAME_ROOT, load_model_and_tokenizer, save_json  # noqa: F401


def prepare_model_from_analysis(
    *,
    analysis_json: str,
    prepare_mode: str,
    prepared_model_dir: str,
    model_path: str | None,
    device: str,
    dtype: str,
) -> str:
    payload = json.loads(Path(analysis_json).read_text())
    resolved_model_path = model_path or payload.get("model_path")
    if not resolved_model_path:
        raise ValueError("`model_path` is required unless it is available in `analysis_json`.")

    model, tokenizer = load_model_and_tokenizer(resolved_model_path, device, dtype=dtype)
    decoder = model.get_decoder()
    decoder.clear_skip_keep_mask()
    if hasattr(decoder, "clear_dynamic_skip_policy"):
        decoder.clear_dynamic_skip_policy()

    if prepare_mode == "best_static":
        keep_mask = payload["best_keep_mask"]
        decoder.set_skip_keep_mask(keep_mask)
    else:
        dynamic = payload.get("dynamic_skip_analysis")
        if dynamic is None:
            raise ValueError("`analysis_json` does not contain `dynamic_skip_analysis`.")
        metric_key = {
            "best_dynamic_ppl": "best_ppl_metrics",
            "best_dynamic_tolerated": "best_tolerated_metrics",
            "best_dynamic_skip": "best_skip_metrics",
        }[prepare_mode]
        metrics = dynamic[metric_key]
        decoder.set_dynamic_skip_policy(
            strategy=dynamic["strategy"],
            position_thresholds=metrics["position_thresholds"],
            max_skips=metrics["max_skips"],
        )

    export_dir = Path(prepared_model_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    save_json(export_dir / "routing_analysis.json", payload)
    return str(export_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm_eval with custom ReSkip flame models.")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--analysis_json", default="")
    parser.add_argument(
        "--prepare_mode",
        choices=("none", "best_static", "best_dynamic_ppl", "best_dynamic_tolerated", "best_dynamic_skip"),
        default="none",
    )
    parser.add_argument(
        "--dynamic_mode",
        choices=("none", "quality", "tolerated", "deepest", "static"),
        default="none",
        help=(
            "User-facing alias for analysis-based model preparation: "
            "'quality' -> best_dynamic_ppl, "
            "'tolerated' -> best_dynamic_tolerated, "
            "'deepest' -> best_dynamic_skip, "
            "'static' -> best_static."
        ),
    )
    parser.add_argument("--prepared_model_dir", default="")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.dynamic_mode != "none":
        alias_map = {
            "quality": "best_dynamic_ppl",
            "tolerated": "best_dynamic_tolerated",
            "deepest": "best_dynamic_skip",
            "static": "best_static",
        }
        if args.prepare_mode != "none":
            raise ValueError("Use either `--prepare_mode` or `--dynamic_mode`, not both.")
        args.prepare_mode = alias_map[args.dynamic_mode]

    resolved_model_path = args.model_path
    if args.prepare_mode != "none":
        if not args.analysis_json:
            raise ValueError("`--analysis_json` is required when `--prepare_mode` is not `none`.")
        if not args.prepared_model_dir:
            raise ValueError("`--prepared_model_dir` is required when `--prepare_mode` is not `none`.")
        resolved_model_path = prepare_model_from_analysis(
            analysis_json=args.analysis_json,
            prepare_mode=args.prepare_mode,
            prepared_model_dir=args.prepared_model_dir,
            model_path=args.model_path or None,
            device=args.device,
            dtype=args.dtype,
        )
    elif not resolved_model_path:
        raise ValueError("`--model_path` is required when `--prepare_mode=none`.")

    model_args = f"pretrained={resolved_model_path},dtype='{args.dtype}'"
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
