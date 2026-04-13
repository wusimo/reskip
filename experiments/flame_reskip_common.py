from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
FLAME_ROOT = REPO_ROOT / "flame"
FLA_ROOT = REPO_ROOT / "flash-linear-attention"

for path in (FLAME_ROOT, FLA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import fla  # noqa: E402,F401
from flame.data import build_dataloader, build_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_bools(text: str) -> list[bool]:
    return [item.strip() in {"1", "true", "True", "yes"} for item in text.split(",") if item.strip()]


def parse_torch_dtype(dtype: str | None, device: str) -> torch.dtype | None:
    if dtype is None or dtype == "auto":
        return torch.bfloat16 if device.startswith("cuda") else None
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]


def count_valid_tokens(labels: torch.Tensor) -> int:
    valid = (labels != -100).sum().item()
    if valid == 0:
        valid = labels.numel()
    return max(int(valid) - labels.size(0), 0)


def build_text_dataloader(
    *,
    tokenizer,
    dataset: str,
    dataset_name: str | None,
    dataset_split: str,
    data_dir: str | None,
    data_files: str | None,
    seq_len: int,
    context_len: int | None,
    batch_size: int,
    num_workers: int,
    streaming: bool,
    varlen: bool,
    seed: int = 0,
):
    ds = build_dataset(
        dataset=dataset,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        data_dir=data_dir,
        data_files=data_files,
        streaming=streaming,
        dp_degree=1,
        num_workers=num_workers,
        seed=seed,
    )
    return build_dataloader(
        dataset=ds,
        tokenizer=tokenizer,
        rank=0,
        world_size=1,
        batch_size=batch_size,
        seq_len=seq_len,
        context_len=context_len,
        varlen=varlen,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        snapshot_every_n_steps=1,
    )


def load_model_and_tokenizer(
    model_path: str,
    device: str,
    dtype: str = "auto",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dtype = parse_torch_dtype(dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def evaluate_causal_lm(
    model,
    dataloader,
    device: str,
    *,
    num_batches: int | None = None,
    return_routing_info: bool = False,
    enable_skipping: bool | None = None,
    skip_keep_mask: list[bool] | None = None,
    return_batch_metrics: bool = False,
    **model_forward_kwargs,
) -> dict[str, Any]:
    total_loss = 0.0
    total_tokens = 0
    total_blocks = 0
    total_compute_units = 0.0
    total_compute_ratio = 0.0
    total_mlp_skips = 0.0
    n_batches = 0
    importance_matrix = None
    block_importance_sum = None
    batch_metrics = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        cu_seqlens = batch.get("cu_seqlens")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            if torch.all(attention_mask):
                attention_mask = None
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            cu_seqlens=cu_seqlens,
            use_cache=False,
            return_dict=True,
            return_routing_info=return_routing_info,
            enable_skipping=enable_skipping,
            skip_keep_mask=skip_keep_mask,
            **model_forward_kwargs,
        )
        valid_tokens = count_valid_tokens(labels)
        total_loss += outputs.loss.item() * valid_tokens
        total_tokens += valid_tokens
        n_batches += 1

        routing_info = getattr(outputs, "routing_info", None)
        if routing_info is not None:
            total_blocks += routing_info["num_blocks_executed"]
            total_compute_units += float(routing_info.get("num_compute_units_executed", routing_info["num_blocks_executed"]))
            total_compute_ratio += float(routing_info.get("compute_ratio", 1.0))
            total_mlp_skips += float(routing_info.get("num_mlp_skipped", 0.0))
            if importance_matrix is None:
                importance_matrix = routing_info["importance_matrix"].float().cpu()
                block_importance_sum = torch.tensor(routing_info["block_importance"], dtype=torch.float32)
            else:
                importance_matrix += routing_info["importance_matrix"].float().cpu()
                block_importance_sum += torch.tensor(routing_info["block_importance"], dtype=torch.float32)
        if return_batch_metrics:
            payload = {
                "loss": outputs.loss.item(),
                "tokens": valid_tokens,
            }
            if routing_info is not None:
                payload["avg_blocks"] = float(routing_info["num_blocks_executed"])
                payload["avg_compute_units"] = float(routing_info.get("num_compute_units_executed", routing_info["num_blocks_executed"]))
                payload["compute_ratio"] = float(routing_info.get("compute_ratio", 1.0))
                payload["num_mlp_skipped"] = float(routing_info.get("num_mlp_skipped", 0.0))
                payload["block_importance"] = list(routing_info["block_importance"])
                payload["execution_trace"] = routing_info.get("execution_trace", [])
                payload["mlp_execution_trace"] = routing_info.get("mlp_execution_trace", [])
            batch_metrics.append(payload)

        if num_batches is not None and n_batches >= num_batches:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    result = {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 20)),
        "tokens": total_tokens,
        "num_batches": n_batches,
    }
    if importance_matrix is not None:
        result["avg_blocks"] = total_blocks / max(n_batches, 1)
        result["avg_compute_units"] = total_compute_units / max(n_batches, 1)
        result["avg_compute_ratio"] = total_compute_ratio / max(n_batches, 1)
        result["avg_mlp_skips"] = total_mlp_skips / max(n_batches, 1)
        result["importance_matrix"] = (importance_matrix / max(n_batches, 1)).tolist()
        result["block_importance"] = (block_importance_sum / max(n_batches, 1)).tolist()
    if return_batch_metrics:
        result["batch_metrics"] = batch_metrics
    return result


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
