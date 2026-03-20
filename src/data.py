"""
Data pipelines for training and evaluation.

Supports:
1. Structured synthetic data (for fast prototyping and unit tests)
2. HuggingFace datasets (WikiText, C4 subset, SlimPajama)
3. Synthetic VLA data with learnable structure
"""

import math
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


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

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
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
