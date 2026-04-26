from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/home/user01/Minko/reskip2/reskip/rechange")
from qwen3vl_attnres_retrofit import Qwen3VLAttnResRetrofit


MODEL_PATH = "/home/user01/Minko/models/Qwen3-VL-2B"
ULTRACHAT_CACHE = "/home/user01/Minko/datasets/ultrachat_200k"
FINEWEB_PATH = "/home/user01/Minko/datasets/fineweb_edu_100BT/data"

IM_START_ID = 151644
IM_END_ID = 151645
ASSISTANT_ID = 77091
NEWLINE_ID = 198
_DATASET_CACHE: dict[str, object] = {}


def _compute_assistant_mask(ids: list[int]) -> list[int]:
    mask = [0] * len(ids)
    i = 0
    n = len(ids)
    while i < n:
        if (
            i + 2 < n
            and ids[i] == IM_START_ID
            and ids[i + 1] == ASSISTANT_ID
            and ids[i + 2] == NEWLINE_ID
        ):
            j = i + 3
            while j < n and ids[j] != IM_END_ID:
                mask[j] = 1
                j += 1
            if j < n:
                mask[j] = 1
                j += 1
            i = j
        else:
            i += 1
    return mask


def build_sft_stream(tok, seq_len: int = 1024, seed: int = 0, split: str = "train_sft"):
    if split not in _DATASET_CACHE:
        _DATASET_CACHE[split] = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            cache_dir=ULTRACHAT_CACHE,
            split=split,
        )
    ds = _DATASET_CACHE[split].shuffle(seed=seed)

    eos_id = tok.eos_token_id if tok.eos_token_id is not None else IM_END_ID
    id_buf: list[int] = []
    lbl_buf: list[int] = []

    for sample in ds:
        messages = sample.get("messages")
        if not messages:
            continue
        try:
            encoded = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        except Exception:
            continue
        mask = _compute_assistant_mask(ids)
        if sum(mask) == 0:
            continue
        labels = [(tok_id if m == 1 else -100) for tok_id, m in zip(ids, mask)]
        id_buf.extend(ids)
        lbl_buf.extend(labels)
        id_buf.append(eos_id)
        lbl_buf.append(-100)
        while len(id_buf) >= seq_len:
            chunk_ids = id_buf[:seq_len]
            chunk_lbl = lbl_buf[:seq_len]
            id_buf = id_buf[seq_len:]
            lbl_buf = lbl_buf[seq_len:]
            yield torch.tensor(chunk_ids, dtype=torch.long), torch.tensor(chunk_lbl, dtype=torch.long)


def build_generic_text_stream(
    tok,
    seq_len: int = 1024,
    seed: int = 0,
    parquet_dir: str = FINEWEB_PATH,
    max_shards: int = 2,
):
    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    rng = random.Random(seed)
    rng.shuffle(parquet_files)
    parquet_files = parquet_files[: max(1, max_shards)]

    eos_id = tok.eos_token_id if tok.eos_token_id is not None else IM_END_ID
    max_text_chars = max(seq_len * 8, 2048)
    id_buf: list[int] = []
    lbl_buf: list[int] = []

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=32, columns=["text"]):
            texts = batch.column("text").to_pylist()
            for text in texts:
                if not text or not isinstance(text, str):
                    continue
                text = text[:max_text_chars]
                try:
                    ids = tok.encode(text, add_special_tokens=False)
                except Exception:
                    continue
                if len(ids) < 8:
                    continue
                labels = list(ids)
                id_buf.extend(ids)
                lbl_buf.extend(labels)
                id_buf.append(eos_id)
                lbl_buf.append(eos_id)
                while len(id_buf) >= seq_len:
                    chunk_ids = id_buf[:seq_len]
                    chunk_lbl = lbl_buf[:seq_len]
                    id_buf = id_buf[seq_len:]
                    lbl_buf = lbl_buf[seq_len:]
                    yield torch.tensor(chunk_ids, dtype=torch.long), torch.tensor(chunk_lbl, dtype=torch.long)


def freeze_for_stage1(model: Qwen3VLAttnResRetrofit):
    model.freeze_base()
    model.freeze_vision()
    for p in model.retrofit_parameters():
        p.requires_grad = True


def unfreeze_late_blocks(model: Qwen3VLAttnResRetrofit):
    for p in model.late_block_parameters():
        p.requires_grad = True


def iter_block_parameters(model: Qwen3VLAttnResRetrofit, block_indices: list[int]):
    params = []
    for block_idx in sorted(set(int(x) for x in block_indices)):
        start = block_idx * model.layers_per_block
        end = start + model.layers_per_block
        for layer_idx in range(start, end):
            params.extend(list(model.text_layers[layer_idx].parameters()))
    return params


def valid_label_mask(labels: torch.Tensor) -> torch.Tensor:
    shifted_labels = torch.cat([labels[..., 1:], torch.full_like(labels[:, :1], -100)], dim=1)
    return (shifted_labels != -100).view(-1)


def compute_masked_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    mask_pos = valid_label_mask(labels)
    vocab_size = student_logits.shape[-1]
    s_flat = student_logits.view(-1, vocab_size)[mask_pos]
    t_flat = teacher_logits.view(-1, vocab_size)[mask_pos]
    if s_flat.shape[0] == 0:
        return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
    s_logp = F.log_softmax(s_flat / temperature, dim=-1)
    t_p = F.softmax(t_flat / temperature, dim=-1)
    return F.kl_div(s_logp.float(), t_p.float(), reduction="batchmean") * (temperature * temperature)


def evaluate(model, teacher, tok, device: str, skip_block: int, num_batches: int, seq_len: int, temperature: float):
    model.eval()
    stream = build_sft_stream(tok, seq_len=seq_len, seed=123, split="test_sft")
    metrics = {"task_loss": 0.0, "full_kl": 0.0, "skip_kl": 0.0, "hidden_loss": 0.0, "n": 0}
    max_attempts = max(num_batches * 32, 32)
    with torch.no_grad():
        for i, (ids_cpu, lbl_cpu) in enumerate(stream):
            if metrics["n"] >= num_batches or i >= max_attempts:
                break
            if (lbl_cpu != -100).sum().item() == 0:
                continue
            ids = ids_cpu.unsqueeze(0).to(device)
            labels = lbl_cpu.unsqueeze(0).to(device)
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            full_out = model(input_ids=ids, labels=labels, return_block_states=True)
            skip_out = model(
                input_ids=ids,
                labels=labels,
                return_block_states=True,
                skip_block_indices=[skip_block],
            )
            hidden_target = full_out.block_outputs[skip_block]
            hidden_pred = skip_out.block_outputs[skip_block]
            metrics["task_loss"] += float(full_out.loss.detach())
            metrics["full_kl"] += float(compute_masked_kl(full_out.logits, teacher_logits, labels, temperature).detach())
            metrics["skip_kl"] += float(compute_masked_kl(skip_out.logits, teacher_logits, labels, temperature).detach())
            metrics["hidden_loss"] += float(F.mse_loss(hidden_pred.float(), hidden_target.float()).detach())
            metrics["n"] += 1
    n = max(metrics["n"], 1)
    return {k: (v / n if k != "n" else v) for k, v in metrics.items()}


def choose_skip_block(model: Qwen3VLAttnResRetrofit, skip_block_arg: int) -> int:
    if skip_block_arg >= 0:
        return skip_block_arg
    return random.choice(list(model.skippable_blocks))


def train(args):
    device = f"cuda:{args.gpu}"
    dtype = torch.bfloat16
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train.log", "a")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tok = processor.tokenizer
    teacher = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    base = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
    attnres_blocks = [int(x) for x in args.attnres_blocks.split(",") if x.strip()]
    model = Qwen3VLAttnResRetrofit(
        base,
        num_blocks=args.num_blocks,
        adapter_rank=args.adapter_rank,
        skippable_blocks=(attnres_blocks or None),
    ).to(device=device, dtype=dtype)
    freeze_for_stage1(model)
    retrofit_params = [p for p in model.retrofit_parameters() if p.requires_grad]
    param_groups = [
        {"params": retrofit_params, "lr": args.lr, "name": "retrofit"},
    ]
    if args.late_block_lr > 0:
        if args.coadapt_blocks:
            target_blocks = [int(x) for x in args.coadapt_blocks.split(",") if x.strip()]
            late_param_source = iter_block_parameters(model, target_blocks)
        else:
            unfreeze_late_blocks(model)
            late_param_source = model.late_block_parameters()
        late_params = []
        retrofit_ids = {id(p) for p in retrofit_params}
        for p in late_param_source:
            p.requires_grad = True
            if id(p) not in retrofit_ids:
                late_params.append(p)
        if late_params:
            param_groups.append({"params": late_params, "lr": args.late_block_lr, "name": "late_block"})
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.0)
    sft_stream = build_sft_stream(tok, seq_len=args.seq_len, seed=args.seed, split="train_sft")
    generic_stream = None
    if args.mix_generic_prob > 0:
        generic_stream = build_generic_text_stream(
            tok,
            seq_len=args.seq_len,
            seed=args.seed,
            parquet_dir=args.generic_parquet_dir,
            max_shards=args.generic_max_shards,
        )

    init_metrics = evaluate(
        model, teacher, tok, device, args.eval_skip_block, args.eval_batches, args.seq_len, args.kd_temperature
    )
    line = f"INIT {init_metrics} gamma_mean={float(model.gamma.detach().float().mean()):.6f}"
    print(line)
    log_f.write(line + "\n")
    log_f.flush()
    best_score = float("inf")
    best_step = 0

    t0 = time.time()
    ema = None
    for step in range(1, args.steps + 1):
        while True:
            use_generic = generic_stream is not None and random.random() < args.mix_generic_prob
            active_stream = generic_stream if use_generic else sft_stream
            try:
                ids_cpu, lbl_cpu = next(active_stream)
            except StopIteration:
                if use_generic:
                    generic_stream = build_generic_text_stream(
                        tok,
                        seq_len=args.seq_len,
                        seed=args.seed + step,
                        parquet_dir=args.generic_parquet_dir,
                        max_shards=args.generic_max_shards,
                    )
                    ids_cpu, lbl_cpu = next(generic_stream)
                else:
                    sft_stream = build_sft_stream(tok, seq_len=args.seq_len, seed=args.seed + step, split="train_sft")
                    ids_cpu, lbl_cpu = next(sft_stream)
            if (lbl_cpu != -100).sum().item() > 0:
                break

        ids = ids_cpu.unsqueeze(0).to(device)
        labels = lbl_cpu.unsqueeze(0).to(device)
        skip_block = choose_skip_block(model, args.skip_block)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits

        model.train()
        full_out = model(input_ids=ids, labels=labels, return_block_states=True)
        skip_out = model(input_ids=ids, labels=labels, return_block_states=True, skip_block_indices=[skip_block])

        hidden_target = full_out.block_outputs[skip_block].detach()
        hidden_pred = skip_out.block_outputs[skip_block]
        hidden_loss = F.mse_loss(hidden_pred.float(), hidden_target.float())
        skip_kl = compute_masked_kl(skip_out.logits, teacher_logits, labels, args.kd_temperature)
        full_kl = compute_masked_kl(full_out.logits, teacher_logits, labels, args.kd_temperature)
        skip_task_loss = skip_out.loss

        entropy_term = torch.zeros((), device=device, dtype=full_out.logits.dtype)
        if full_out.entropy_penalty is not None and args.entropy_weight > 0:
            entropy_term = args.entropy_weight * full_out.entropy_penalty

        total = (
            full_out.loss
            + args.skip_task_weight * skip_task_loss
            + args.hidden_weight * hidden_loss
            + args.skip_kl_weight * skip_kl
            + args.full_kl_weight * full_kl
            + entropy_term
        )
        opt.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        opt.step()

        current = {
            "task": float(full_out.loss.detach()),
            "skip_task": float(skip_task_loss.detach()),
            "hidden": float(hidden_loss.detach()),
            "skip_kl": float(skip_kl.detach()),
            "full_kl": float(full_kl.detach()),
            "total": float(total.detach()),
        }
        if ema is None:
            ema = current.copy()
        else:
            for key in ema:
                ema[key] = 0.98 * ema[key] + 0.02 * current[key]

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t0
            tps = step * args.seq_len / max(elapsed, 1e-6)
            line = (
                f"step {step}/{args.steps} "
                f"task={ema['task']:.4f} skip_task={ema['skip_task']:.4f} hidden={ema['hidden']:.4f} "
                f"skip_kl={ema['skip_kl']:.4f} full_kl={ema['full_kl']:.4f} "
                f"total={ema['total']:.4f} gamma_mean={float(model.gamma.detach().float().mean()):.6f} "
                f"skip_block={skip_block} grad_norm={float(grad_norm.detach().float()):.6f} "
                f"tps={tps:,.0f}"
            )
            print(line)
            log_f.write(line + "\n")
            log_f.flush()

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model, teacher, tok, device, args.eval_skip_block, args.eval_batches, args.seq_len, args.kd_temperature
            )
            line = f"EVAL step={step} {metrics} gamma_mean={float(model.gamma.detach().float().mean()):.6f}"
            print(line)
            log_f.write(line + "\n")
            log_f.flush()
            if metrics["full_kl"] <= args.best_full_kl_max:
                score = metrics["skip_kl"] + args.best_hidden_weight * metrics["hidden_loss"]
                if score < best_score:
                    best_score = score
                    best_step = step
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": vars(args),
                            "gamma": model.gamma.detach().cpu(),
                            "skippable_blocks": list(model.skippable_blocks),
                            "eval_metrics": metrics,
                            "best_step": best_step,
                            "best_score": best_score,
                        },
                        out_dir / "attnres_retrofit_stage1_best.pt",
                    )
                    best_line = (
                        f"BEST step={step} score={best_score:.6f} "
                        f"full_kl={metrics['full_kl']:.6f} skip_kl={metrics['skip_kl']:.6f} "
                        f"hidden={metrics['hidden_loss']:.6f}"
                    )
                    print(best_line)
                    log_f.write(best_line + "\n")
                    log_f.flush()

    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "gamma": model.gamma.detach().cpu(),
            "skippable_blocks": list(model.skippable_blocks),
        },
        out_dir / "attnres_retrofit_stage1.pt",
    )
    final_line = f"FINAL_BEST step={best_step} score={best_score if best_score < float('inf') else None}"
    print(final_line)
    log_f.write(final_line + "\n")
    log_f.flush()
    log_f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=14)
    parser.add_argument("--adapter-rank", type=int, default=128)
    parser.add_argument("--attnres-blocks", default="", help="comma-separated block ids using AttnRes injection")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-block", type=int, default=10)
    parser.add_argument("--eval-skip-block", type=int, default=10)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--skip-task-weight", type=float, default=0.0)
    parser.add_argument("--skip-kl-weight", type=float, default=1.0)
    parser.add_argument("--full-kl-weight", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--late-block-lr", type=float, default=0.0)
    parser.add_argument("--coadapt-blocks", default="", help="comma-separated block ids to unfreeze with late-block-lr")
    parser.add_argument("--mix-generic-prob", type=float, default=0.0)
    parser.add_argument("--generic-parquet-dir", default=FINEWEB_PATH)
    parser.add_argument("--generic-max-shards", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--best-full-kl-max", type=float, default=0.05)
    parser.add_argument("--best-hidden-weight", type=float, default=0.001)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
