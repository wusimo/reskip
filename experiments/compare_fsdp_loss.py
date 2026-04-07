import argparse
import os

import fla  # noqa: F401
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models  # noqa: F401
from flame.data import build_dataloader, build_dataset
from flame.models.parallelize_fla import apply_fsdp
from fla.ops.utils import prepare_position_ids


def build_batch(tokenizer, batch_size, seq_len, context_len, dataset_path, seed):
    dataset = build_dataset(
        dataset=dataset_path,
        dataset_name=None,
        dataset_split="train",
        data_dir=None,
        data_files=None,
        data_probs=None,
        streaming=True,
        dp_degree=1,
        num_workers=0,
        seed=seed,
    )
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=0,
        world_size=1,
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

    batch = build_batch(
        tokenizer=tokenizer,
        batch_size=1,
        seq_len=args.seq_len,
        context_len=args.context_len,
        dataset_path=args.dataset_path,
        seed=args.seed,
    )

    if rank == 0:
        input_ids = batch["input_ids"].contiguous().to(device)
        labels = batch["labels"].contiguous().to(device)
        cu_seqlens = batch["cu_seqlens"].contiguous().to(device)
    else:
        input_ids = torch.empty_like(batch["input_ids"], device=device)
        labels = torch.empty_like(batch["labels"], device=device)
        cu_seqlens = torch.empty_like(batch["cu_seqlens"], device=device)

    dist.broadcast(input_ids, src=0)
    dist.broadcast(labels, src=0)
    dist.broadcast(cu_seqlens, src=0)

    position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)

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
    model.eval()

    state = {"model": get_model_state_dict(model)}
    checkpoint_id = os.path.join(args.checkpoint_dir, f"step-{args.step}")
    dcp.load(state, checkpoint_id=checkpoint_id)
    set_model_state_dict(
        model,
        model_state_dict=state["model"],
        options=StateDictOptions(strict=False),
    )

    position_ids = position_ids.to(device)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            position_ids=position_ids,
            use_cache=False,
        )
        loss = output.loss.detach().float()

    loss_list = [torch.zeros_like(loss) for _ in range(world_size)]
    dist.all_gather(loss_list, loss)

    if rank == 0:
        values = [float(v.item()) for v in loss_list]
        print(f"checkpoint_step={args.step}")
        print(f"rank_losses={values}")
        print(f"mean_loss={sum(values) / len(values):.6f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
