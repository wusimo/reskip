import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
FLAME_ROOT = REPO_ROOT / "flame"
FLA_ROOT = REPO_ROOT / "flash-linear-attention"

for path in (FLAME_ROOT, FLA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import fla  # noqa: E402,F401
import custom_models  # noqa: F401
from flame.data import build_dataloader, build_dataset
from flame.models.parallelize_fla import apply_fsdp
from fla.ops.utils import prepare_position_ids


def build_batch(
    tokenizer,
    batch_size,
    seq_len,
    context_len,
    dataset_path,
    seed,
    dp_degree,
    dp_rank,
):
    dataset = build_dataset(
        dataset=dataset_path,
        dataset_name=None,
        dataset_split="train",
        data_dir=None,
        data_files=None,
        data_probs=None,
        streaming=True,
        dp_degree=dp_degree,
        num_workers=0,
        seed=seed,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=batch_size,
        seq_len=seq_len,
        context_len=context_len,
        varlen=True,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        snapshot_every_n_steps=0,
    )
    return next(iter(dataloader))


def build_eval_dataloader(
    tokenizer,
    batch_size,
    seq_len,
    context_len,
    dataset_path,
    seed,
    dp_degree,
    dp_rank,
):
    dataset = build_dataset(
        dataset=dataset_path,
        dataset_name=None,
        dataset_split="train",
        data_dir=None,
        data_files=None,
        data_probs=None,
        streaming=True,
        dp_degree=dp_degree,
        num_workers=0,
        seed=seed,
    )
    return build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=batch_size,
        seq_len=seq_len,
        context_len=context_len,
        varlen=True,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        snapshot_every_n_steps=0,
    )


def build_stateful_dataloader(
    tokenizer,
    batch_size,
    seq_len,
    context_len,
    dataset_path,
    seed,
    dp_degree,
    dp_rank,
    num_workers,
    snapshot_every_n_steps,
):
    dataset = build_dataset(
        dataset=dataset_path,
        dataset_name=None,
        dataset_split="train",
        data_dir=None,
        data_files=None,
        data_probs=None,
        streaming=True,
        dp_degree=dp_degree,
        num_workers=num_workers,
        seed=seed,
    )
    return build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=batch_size,
        seq_len=seq_len,
        context_len=context_len,
        varlen=True,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False,
        snapshot_every_n_steps=snapshot_every_n_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=32768)
    parser.add_argument("--context_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_parallel_degree", type=int, default=None)
    parser.add_argument("--data_parallel_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--snapshot_every_n_steps", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--use_checkpoint_dataloader_state", action="store_true")
    parser.add_argument("--train_mode", action="store_true")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    dp_degree = args.data_parallel_degree or world_size

    checkpoint_id = os.path.join(args.checkpoint_dir, f"step-{args.step}")
    stateful_dataloader = None
    eval_dataloader = None
    if args.use_checkpoint_dataloader_state:
        stateful_dataloader = build_stateful_dataloader(
            tokenizer=tokenizer,
            batch_size=1,
            seq_len=args.seq_len,
            context_len=args.context_len,
            dataset_path=args.dataset_path,
            seed=args.seed,
            dp_degree=dp_degree,
            dp_rank=args.data_parallel_rank,
            num_workers=args.num_workers,
            snapshot_every_n_steps=args.snapshot_every_n_steps,
        )
        dcp.load({"dataloader": stateful_dataloader}, checkpoint_id=checkpoint_id)
        data_iterator = iter(stateful_dataloader)
    else:
        eval_dataloader = build_eval_dataloader(
            tokenizer=tokenizer,
            batch_size=1,
            seq_len=args.seq_len,
            context_len=args.context_len,
            dataset_path=args.dataset_path,
            seed=args.seed,
            dp_degree=dp_degree,
            dp_rank=args.data_parallel_rank,
        )
        data_iterator = iter(eval_dataloader)

    config = AutoConfig.from_pretrained(args.config_path, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard_cp",))
    apply_fsdp(
        model,
        dp_mesh=mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        cpu_offload=False,
        reshard_after_forward_policy="default",
    )
    model.to_empty(device=device)
    with torch.no_grad():
        model.post_init()
    if args.train_mode:
        model.train()
    else:
        model.eval()

    state = {"model": get_model_state_dict(model)}
    dcp.load(state, checkpoint_id=checkpoint_id)
    set_model_state_dict(
        model,
        model_state_dict=state["model"],
        options=StateDictOptions(strict=False),
    )

    micro_losses = []
    batch_losses = []
    grad_ctx = torch.enable_grad if args.train_mode else torch.no_grad
    with grad_ctx():
        for _ in range(args.num_batches):
            if args.use_checkpoint_dataloader_state:
                batch = next(data_iterator)
            else:
                batch = next(data_iterator)

            input_ids = batch["input_ids"].contiguous().to(device)
            labels = batch["labels"].contiguous().to(device)
            cu_seqlens = batch["cu_seqlens"].contiguous().to(device)

            current_micro_losses = []
            for _ in range(args.grad_accum_steps):
                position_ids = prepare_position_ids(cu_seqlens).to(torch.int32).to(device)
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    cu_seqlens=cu_seqlens,
                    position_ids=position_ids,
                    use_cache=False,
                )
                micro_loss = output.loss.detach().float()
                micro_losses.append(micro_loss)
                current_micro_losses.append(micro_loss)
            batch_losses.append(torch.stack(current_micro_losses).mean())
        loss = torch.stack(batch_losses).mean()

    loss_list = [torch.zeros_like(loss) for _ in range(world_size)]
    dist.all_gather(loss_list, loss)

    if rank == 0:
        values = [float(v.item()) for v in loss_list]
        micro_values = [float(v.item()) for v in micro_losses]
        print(f"checkpoint_step={args.step}")
        print(f"data_parallel_degree={dp_degree}")
        print(f"data_parallel_rank={args.data_parallel_rank}")
        print(f"use_checkpoint_dataloader_state={args.use_checkpoint_dataloader_state}")
        print(f"grad_accum_steps={args.grad_accum_steps}")
        print(f"num_batches={args.num_batches}")
        print(f"train_mode={args.train_mode}")
        print(f"rank0_micro_losses={micro_values}")
        print(f"rank0_batch_losses={[float(v.item()) for v in batch_losses]}")
        print(f"rank_losses={values}")
        print(f"mean_loss={sum(values) / len(values):.6f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
