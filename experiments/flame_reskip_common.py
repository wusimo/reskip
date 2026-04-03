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


def count_valid_tokens(labels: torch.Tensor) -> int:
    valid = (labels != -100).sum().item()
    if valid == 0:
        valid = labels.numel()
    return int(valid)


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
        seed=42,
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
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
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
) -> dict[str, Any]:
    total_loss = 0.0
    total_tokens = 0
    total_blocks = 0
    n_batches = 0
    importance_matrix = None
    block_importance_sum = None

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            return_routing_info=return_routing_info,
            enable_skipping=enable_skipping,
            skip_keep_mask=skip_keep_mask,
        )
        valid_tokens = count_valid_tokens(labels)
        total_loss += outputs.loss.item() * valid_tokens
        total_tokens += valid_tokens
        n_batches += 1

        routing_info = getattr(outputs, "routing_info", None)
        if routing_info is not None:
            total_blocks += routing_info["num_blocks_executed"]
            if importance_matrix is None:
                importance_matrix = routing_info["importance_matrix"].float().cpu()
                block_importance_sum = torch.tensor(routing_info["block_importance"], dtype=torch.float32)
            else:
                importance_matrix += routing_info["importance_matrix"].float().cpu()
                block_importance_sum += torch.tensor(routing_info["block_importance"], dtype=torch.float32)

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
        result["importance_matrix"] = (importance_matrix / max(n_batches, 1)).tolist()
        result["block_importance"] = (block_importance_sum / max(n_batches, 1)).tolist()
    return result


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
