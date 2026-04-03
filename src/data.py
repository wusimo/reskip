"""
Data pipelines for training and evaluation.

Supports:
1. Structured synthetic data (for fast prototyping and unit tests)
2. HuggingFace datasets (WikiText)
3. Local text/JSONL corpora streamed from disk for large-scale LM training
4. Synthetic VLA data with learnable structure
"""

import gzip
import json
import math
import os
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Language Modeling Datasets
# ---------------------------------------------------------------------------


class StructuredSyntheticLM(Dataset):
    """
    Synthetic LM data with learnable structure.

    Unlike pure random tokens, this generates data with patterns:
    - Copying: output repeats a prefix
    - Counting: simple arithmetic sequences
    - Lookup: key-value associations
    - Noise: random padding

    This lets us verify that the model actually learns, and that
    routing patterns differ by input difficulty.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        seq_len: int = 256,
        num_samples: int = 50000,
        difficulty_mix: str = "mixed",  # "easy", "hard", "mixed"
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.difficulty_mix = difficulty_mix

        rng = torch.Generator().manual_seed(seed)

        # Reserve special tokens
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.SEP = 3
        token_range = (4, min(vocab_size, 512))  # Use limited vocab for structure

        self.data = []
        self.labels = []
        self.difficulties = []

        for i in range(num_samples):
            if difficulty_mix == "easy":
                diff = 0
            elif difficulty_mix == "hard":
                diff = 2
            else:
                diff = i % 3  # Rotate: 0=easy, 1=medium, 2=hard

            tokens, target = self._generate_sample(diff, seq_len, token_range, rng)
            self.data.append(tokens)
            self.labels.append(target)
            self.difficulties.append(diff)

        self.data = torch.stack(self.data)
        self.labels = torch.stack(self.labels)
        self.difficulties = torch.tensor(self.difficulties)

    def _generate_sample(self, difficulty, seq_len, token_range, rng):
        lo, hi = token_range

        if difficulty == 0:
            # Easy: repeat a short pattern
            pattern_len = torch.randint(3, 8, (1,), generator=rng).item()
            pattern = torch.randint(lo, hi, (pattern_len,), generator=rng)
            repeats = math.ceil(seq_len / pattern_len)
            tokens = pattern.repeat(repeats)[:seq_len]

        elif difficulty == 1:
            # Medium: copy prefix after separator
            prefix_len = seq_len // 3
            prefix = torch.randint(lo, hi, (prefix_len,), generator=rng)
            sep = torch.tensor([self.SEP])
            suffix = prefix.clone()
            padding = torch.randint(lo, hi, (seq_len - prefix_len * 2 - 1,), generator=rng)
            tokens = torch.cat([prefix, sep, suffix, padding])[:seq_len]

        else:
            # Hard: multi-step computation (simple arithmetic mod vocab)
            base = torch.randint(lo, hi, (seq_len // 4,), generator=rng)
            step = torch.randint(1, 5, (1,), generator=rng).item()
            computed = []
            for b in base:
                computed.extend([b.item(), (b.item() + step) % hi, (b.item() + 2*step) % hi, (b.item() + 3*step) % hi])
            tokens = torch.tensor(computed[:seq_len], dtype=torch.long)

        # Labels = shifted tokens (next-token prediction)
        labels = tokens.clone()
        return tokens, labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx],
            "labels": self.labels[idx],
            "difficulty": self.difficulties[idx],
        }


class HuggingFaceLMDataset(Dataset):
    """
    Wrapper around HuggingFace datasets for real language modeling data.

    Tokenizes and chunks text into fixed-length sequences.
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        split: str = "train",
        seq_len: int = 256,
        tokenizer_name: str = "gpt2",
        max_samples: Optional[int] = None,
    ):
        from datasets import load_dataset
        import tiktoken

        self.seq_len = seq_len

        # Load tokenizer
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.enc.n_vocab

        # Load and tokenize dataset
        print(f"Loading {dataset_name}/{dataset_config} ({split})...")
        ds = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)

        # Tokenize all text
        all_tokens = []
        for example in ds:
            text = example.get("text", "")
            if text.strip():
                tokens = self.enc.encode(text)
                all_tokens.extend(tokens)

            if max_samples and len(all_tokens) >= max_samples * seq_len:
                break

        # Chunk into sequences
        total_tokens = len(all_tokens)
        n_sequences = total_tokens // seq_len
        if max_samples:
            n_sequences = min(n_sequences, max_samples)

        self.data = torch.tensor(
            all_tokens[:n_sequences * seq_len], dtype=torch.long
        ).reshape(n_sequences, seq_len)

        print(f"Created {n_sequences} sequences of length {seq_len} "
              f"(vocab_size={self.vocab_size})")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return {"input_ids": tokens, "labels": tokens}


def _load_tiktoken_tokenizer(tokenizer_name: str):
    import tiktoken

    aliases = {
        "gpt-neox": "gpt2",
        "gpt_neox": "gpt2",
        "gpt-neox-20b": "gpt2",
    }
    resolved_name = aliases.get(tokenizer_name, tokenizer_name)

    try:
        enc = tiktoken.get_encoding(resolved_name)
    except KeyError:
        enc = tiktoken.encoding_for_model(resolved_name)
    return enc


class LocalTextLMDataset(IterableDataset):
    """
    Stream text from local `.txt`, `.jsonl`, `.json`, `.parquet`, and `.gz` files.

    Each yielded example is a fixed-length token chunk suitable for causal LM.
    Sharding is handled inside the dataset so it can be used under DDP without
    requiring a distributed sampler.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 2048,
        tokenizer_name: str = "gpt2",
        max_samples: Optional[int] = None,
        text_key: str = "text",
        global_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.data_path = data_path
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.text_key = text_key
        self.global_rank = global_rank
        self.world_size = world_size

        self.enc = _load_tiktoken_tokenizer(tokenizer_name)
        self.vocab_size = self.enc.n_vocab
        self.files = self._discover_files(data_path)
        if not self.files:
            raise FileNotFoundError(f"No supported text files found under: {data_path}")

    def _discover_files(self, data_path: str) -> list[str]:
        if os.path.isfile(data_path):
            return [data_path]

        supported = (
            ".txt", ".text", ".jsonl", ".json", ".parquet",
            ".txt.gz", ".jsonl.gz", ".json.gz",
        )
        files = []
        for root, _, filenames in os.walk(data_path):
            for name in sorted(filenames):
                if name.endswith(supported):
                    files.append(os.path.join(root, name))
        return sorted(files)

    def _iter_parquet_texts(self, path: str) -> Iterable[str]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Reading local parquet datasets requires `pyarrow`. "
                "Install it with `pip install pyarrow`."
            ) from exc

        parquet_file = pq.ParquetFile(path)
        schema_names = set(parquet_file.schema.names)
        if self.text_key not in schema_names:
            raise KeyError(
                f"Column `{self.text_key}` not found in parquet file {path}. "
                f"Available columns: {sorted(schema_names)}"
            )

        for row_group_idx in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_idx, columns=[self.text_key])
            column = table.column(self.text_key)
            for value in column.to_pylist():
                if isinstance(value, str) and value.strip():
                    yield value

    def _open_file(self, path: str):
        if path.endswith(".gz"):
            return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
        return open(path, "r", encoding="utf-8", errors="ignore")

    def _iter_texts(self) -> Iterable[str]:
        for path in self.files:
            if path.endswith(".parquet"):
                yield from self._iter_parquet_texts(path)
                continue

            with self._open_file(path) as f:
                if path.endswith((".jsonl", ".jsonl.gz")):
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = record.get(self.text_key, "")
                        if text:
                            yield text
                elif path.endswith((".json", ".json.gz")):
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        text = data.get(self.text_key, "")
                        if text:
                            yield text
                    elif isinstance(data, list):
                        for record in data:
                            if isinstance(record, dict):
                                text = record.get(self.text_key, "")
                                if text:
                                    yield text
                else:
                    for line in f:
                        text = line.strip()
                        if text:
                            yield text

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        total_shards = max(1, self.world_size * num_workers)
        shard_id = self.global_rank * num_workers + worker_id

        buffer = []
        emitted = 0
        sample_idx = 0

        for text in self._iter_texts():
            tokens = self.enc.encode_ordinary(text)
            if not tokens:
                continue
            buffer.extend(tokens + [self.enc.eot_token])

            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                del buffer[:self.seq_len]

                if sample_idx % total_shards == shard_id:
                    sample = torch.tensor(chunk, dtype=torch.long)
                    yield {"input_ids": sample, "labels": sample.clone()}
                    emitted += 1
                    if self.max_samples is not None and emitted >= self.max_samples:
                        return
                sample_idx += 1

    def __len__(self):
        if self.max_samples is None:
            raise TypeError("Streaming dataset length is undefined without max_samples")
        return self.max_samples


# ---------------------------------------------------------------------------
# VLA Datasets
# ---------------------------------------------------------------------------


class StructuredVLADataset(Dataset):
    """
    Synthetic VLA dataset with meaningful structure.

    Generates vision-language-action triplets where actions
    have learnable relationships to vision and language inputs:
    - "reach": action = linear function of target position in vision
    - "push": action = direction from current to target + force
    - "pick-place": multi-phase action sequence (reach, grasp, lift, place)

    The difficulty determines how many phases the action sequence has,
    testing whether deeper models/more loops help complex tasks.
    """

    def __init__(
        self,
        vision_dim: int = 1024,
        vision_seq_len: int = 64,
        lang_seq_len: int = 32,
        action_dim: int = 7,
        action_chunk_size: int = 16,
        vocab_size: int = 32000,
        num_samples: int = 5000,
        task_type: str = "mixed",  # "reach", "push", "pick_place", "mixed"
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.task_type = task_type

        rng = torch.Generator().manual_seed(seed)

        self.vision_features = torch.randn(num_samples, vision_seq_len, vision_dim, generator=rng) * 0.1
        self.input_ids = torch.randint(0, vocab_size, (num_samples, lang_seq_len), generator=rng)
        self.target_actions = torch.zeros(num_samples, action_chunk_size, action_dim)
        self.task_labels = []

        for i in range(num_samples):
            if task_type == "reach" or (task_type == "mixed" and i % 3 == 0):
                task = "reach"
            elif task_type == "push" or (task_type == "mixed" and i % 3 == 1):
                task = "push"
            else:
                task = "pick_place"

            self.task_labels.append(task)

            # Extract "target position" from vision (mean of first few patches)
            target = self.vision_features[i, :4, :action_dim].mean(dim=0)
            # Extract "current position" from vision (mean of next patches)
            current = self.vision_features[i, 4:8, :action_dim].mean(dim=0)

            if task == "reach":
                # Simple: linear interpolation to target
                for t in range(action_chunk_size):
                    alpha = (t + 1) / action_chunk_size
                    self.target_actions[i, t, :3] = current[:3] * (1 - alpha) + target[:3] * alpha
                    self.target_actions[i, t, 3:6] = 0.0  # No rotation
                    self.target_actions[i, t, 6] = 0.0     # Gripper open

            elif task == "push":
                # Medium: move to object then push in direction
                direction = (target[:3] - current[:3])
                direction = direction / (direction.norm() + 1e-6)
                for t in range(action_chunk_size):
                    if t < action_chunk_size // 2:
                        # Approach
                        alpha = t / (action_chunk_size // 2)
                        self.target_actions[i, t, :3] = current[:3] + (target[:3] - current[:3]) * alpha
                    else:
                        # Push
                        push_t = (t - action_chunk_size // 2) / (action_chunk_size // 2)
                        self.target_actions[i, t, :3] = target[:3] + direction * push_t * 0.1
                    self.target_actions[i, t, 6] = 0.0

            else:  # pick_place
                # Hard: reach, grasp, lift, move, place
                phases = 5
                phase_len = action_chunk_size // phases
                for t in range(action_chunk_size):
                    phase = min(t // max(phase_len, 1), phases - 1)
                    phase_progress = (t % max(phase_len, 1)) / max(phase_len, 1)

                    if phase == 0:  # Reach
                        self.target_actions[i, t, :3] = current[:3] + (target[:3] - current[:3]) * phase_progress
                        self.target_actions[i, t, 6] = 1.0  # Open
                    elif phase == 1:  # Grasp
                        self.target_actions[i, t, :3] = target[:3]
                        self.target_actions[i, t, 6] = 1.0 - phase_progress  # Close
                    elif phase == 2:  # Lift
                        self.target_actions[i, t, :3] = target[:3]
                        self.target_actions[i, t, 2] += phase_progress * 0.1
                        self.target_actions[i, t, 6] = 0.0
                    elif phase == 3:  # Move
                        place_pos = -target[:3]  # Move to opposite side
                        self.target_actions[i, t, :3] = target[:3] + (place_pos - target[:3]) * phase_progress
                        self.target_actions[i, t, 2] += 0.1
                        self.target_actions[i, t, 6] = 0.0
                    else:  # Place
                        self.target_actions[i, t, :3] = -target[:3]
                        self.target_actions[i, t, 2] -= phase_progress * 0.1
                        self.target_actions[i, t, 6] = phase_progress  # Open

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "vision_features": self.vision_features[idx],
            "input_ids": self.input_ids[idx],
            "target_actions": self.target_actions[idx],
            "task_type": self.task_labels[idx],
        }


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------


def create_lm_dataloaders(
    dataset_type: str = "structured_synthetic",
    batch_size: int = 32,
    seq_len: int = 256,
    vocab_size: int = 32000,
    num_train: int = 50000,
    num_val: int = 5000,
    data_path: str = "",
    val_data_path: str = "",
    tokenizer_name: str = "gpt2",
    text_key: str = "text",
    global_rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    **kwargs,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Create train/val dataloaders.

    Returns: (train_loader, val_loader, actual_vocab_size)
    """
    if dataset_type == "structured_synthetic":
        train_ds = StructuredSyntheticLM(
            vocab_size=vocab_size, seq_len=seq_len,
            num_samples=num_train, difficulty_mix="mixed", seed=42,
        )
        val_ds = StructuredSyntheticLM(
            vocab_size=vocab_size, seq_len=seq_len,
            num_samples=num_val, difficulty_mix="mixed", seed=123,
        )
        actual_vocab = vocab_size

    elif dataset_type == "wikitext":
        train_ds = HuggingFaceLMDataset(
            dataset_name="wikitext", dataset_config="wikitext-103-raw-v1",
            split="train", seq_len=seq_len, max_samples=num_train,
        )
        val_ds = HuggingFaceLMDataset(
            dataset_name="wikitext", dataset_config="wikitext-103-raw-v1",
            split="validation", seq_len=seq_len, max_samples=num_val,
        )
        actual_vocab = train_ds.vocab_size

    elif dataset_type == "wikitext2":
        train_ds = HuggingFaceLMDataset(
            dataset_name="wikitext", dataset_config="wikitext-2-raw-v1",
            split="train", seq_len=seq_len, max_samples=num_train,
        )
        val_ds = HuggingFaceLMDataset(
            dataset_name="wikitext", dataset_config="wikitext-2-raw-v1",
            split="validation", seq_len=seq_len, max_samples=num_val,
        )
        actual_vocab = train_ds.vocab_size

    elif dataset_type in {"local_text", "slimpajama"}:
        if not data_path:
            raise ValueError(f"`data_path` is required for dataset_type={dataset_type}")
        per_rank_train = math.ceil(num_train / max(world_size, 1))
        train_ds = LocalTextLMDataset(
            data_path=data_path,
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            max_samples=per_rank_train,
            text_key=text_key,
            global_rank=global_rank,
            world_size=world_size,
        )
        val_ds = LocalTextLMDataset(
            data_path=val_data_path or data_path,
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            max_samples=num_val,
            text_key=text_key,
            global_rank=0,
            world_size=1,
        )
        actual_vocab = train_ds.vocab_size

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_sampler = None
    val_sampler = None
    if world_size > 1 and not isinstance(train_ds, IterableDataset):
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=global_rank, shuffle=True)
    if world_size > 1 and not isinstance(val_ds, IterableDataset):
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=global_rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None and isinstance(train_ds, Dataset) and not isinstance(train_ds, IterableDataset),
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=num_workers,
    )

    return train_loader, val_loader, actual_vocab


def create_vla_dataloaders(
    batch_size: int = 16,
    num_train: int = 5000,
    num_val: int = 1000,
    task_type: str = "mixed",
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Create VLA train/val dataloaders."""
    train_ds = StructuredVLADataset(
        num_samples=num_train, task_type=task_type, seed=42, **kwargs,
    )
    val_ds = StructuredVLADataset(
        num_samples=num_val, task_type=task_type, seed=123, **kwargs,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    return train_loader, val_loader
