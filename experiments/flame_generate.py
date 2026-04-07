from __future__ import annotations

import argparse
import json

import torch

from flame_reskip_common import (
    load_model_and_tokenizer,
    parse_csv_bools,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a flame ReSkip checkpoint.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--keep_mask", default="")
    parser.add_argument("--routing_analysis", default="")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, dtype=args.dtype)
    decoder = model.get_decoder()
    if args.keep_mask:
        decoder.set_skip_keep_mask(parse_csv_bools(args.keep_mask))
    elif args.routing_analysis:
        with open(args.routing_analysis) as f:
            payload = json.load(f)
        decoder.set_skip_keep_mask(payload["best_keep_mask"])

    batch = tokenizer(args.prompt, return_tensors="pt")
    batch = {key: value.to(args.device) for key, value in batch.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
