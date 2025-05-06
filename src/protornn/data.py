import random
from os import PathLike
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from protornn.tokenizer import ProteinTokenizer


class SequenceDataset(Dataset):
    """Dataset of string sequences."""

    def __init__(self, sequences: list[str], sort: bool = False) -> None:
        if sort:
            sequences = sorted(sequences, key=len, reverse=True)
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx) -> str:
        return self.sequences[idx]


def read_fasta(fasta_path: str | PathLike) -> list[str]:
    """Read sequences from a FASTA file."""
    sequences = []
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

    if current_seq:
        sequences.append("".join(current_seq))

    return sequences


class BatchSampler(Sampler):
    """Batch sampler that returns contiguous batches of data.

    Args:
        dataset_size: Number of items in dataset
        batch_size: Number of items per batch
        drop_last: Drop last batch if incomplete
        shuffle: If True, shuffle batch order (maintain order in batch)

    Examples:
        >>> list(ContiguousBatchSampler(10, 3, shuffle=False, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(ContiguousBatchSampler(10, 3, shuffle=False, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        >>> list(ContiguousBatchSampler(10, 3, shuffle=True, drop_last=True))
        [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    """

    def __init__(self, dataset_size, batch_size, drop_last=False, shuffle=True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.starts = list(range(0, dataset_size, batch_size))
        if drop_last and dataset_size % batch_size:
            self.starts.pop()

    def __len__(self):
        return len(self.starts)

    def __iter__(self):
        if self.shuffle:
            starts = self.starts.copy()
            random.shuffle(starts)
        else:
            starts = self.starts

        for idx in starts:
            end = min(idx + self.batch_size, self.dataset_size)
            yield list(range(idx, end))


def create_collate_fn(tokenizer: ProteinTokenizer, max_length: int | None = None):
    """Create a collate function for protein sequences.

    Args:
        tokenizer: Tokenizer for protein sequences
        max_length: If provided, pad all batches to this length.
            Otherwise, pad to longest sequence in batch.
    """

    def collate_fn(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max_length or max(map(len, sequences), default=0)

        inputs = tokenizer.batch_encode(
            sequences,
            add_bos=True,
            add_eos=False,
            pad_to=max_len + 1,
        )
        targets = tokenizer.batch_encode(
            sequences,
            add_bos=False,
            add_eos=True,
            pad_to=max_len + 1,
        )
        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )

    return collate_fn


def create_dataloaders(
    fasta_path: str | PathLike,
    tokenizer: ProteinTokenizer,
    batch_size: int = 32,
    val_split: float = 0.1,
    test_split: float = 0.1,
    min_length: int = 10,
    max_length: int = 512,
    pack_sequences: bool = True,
    sample: float = 1.0,
    seed: int = 2137,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders from a FASTA file.

    Args:
        fasta_path: Path to FASTA file containing protein sequences
        tokenizer: Tokenizer for encoding protein sequences
        batch_size: Number of sequences per batch
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        min_length: Minimum sequence length to include
        max_length: Maximum sequence length to include
        batch_by_length: If True, use length-based batching strategy
        sample: Fraction of total sequences to use
        seed: Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader, test_loader)

    Note:
        Train dataset is always sorted by descencing sequence length.
        Use `pack_sequences` to minimize padding across batches.

        When `pack_sequences` is True:
            - keep sequence order
            - shuffle batches
            - pad to longest sequence in batch

        When `pack_sequences` is False:
            - shuffle sequences
            - pad to max_length
    """
    # Read sequences
    sequences = read_fasta(fasta_path)

    # Filter out sequences that are too long or too short
    sequences = [s for s in sequences if len(s) <= max_length and len(s) >= min_length]

    rng = torch.manual_seed(seed)
    sample_size = int(len(sequences) * sample)
    indices = torch.randperm(len(sequences), generator=rng).tolist()[:sample_size]

    # Split into train, validation, and test
    val_size = int(sample_size * val_split)
    test_size = int(sample_size * test_split)
    train_size = sample_size - val_size - test_size

    train_sequences = [sequences[i] for i in indices[:train_size]]
    val_sequences = [sequences[i] for i in indices[train_size : train_size + val_size]]
    test_sequences = [sequences[i] for i in indices[train_size + val_size :]]

    # Create datasets
    train_dataset = SequenceDataset(train_sequences, sort=True)
    val_dataset = SequenceDataset(val_sequences)
    test_dataset = SequenceDataset(test_sequences)

    if pack_sequences:
        # length-ordered Dataset -> minimal batch padding
        collate_fn = create_collate_fn(tokenizer)
        train_batch_config = dict(
            batch_sampler=BatchSampler(len(train_dataset), batch_size),
        )
    else:
        collate_fn = create_collate_fn(tokenizer, max_length)
        train_batch_config = dict(
            batch_size=batch_size,
            shuffle=True,
        )

    train_config: dict[str, Any] = dict(
        collate_fn=collate_fn, pin_memory=True, **train_batch_config
    )
    eval_config: dict[str, Any] = dict(
        collate_fn=collate_fn, pin_memory=True, batch_size=batch_size, shuffle=False
    )
    # Train: shuffle batches, not items
    train_loader = DataLoader(train_dataset, **train_config)
    # Val, test: standard sampler, no shuffling
    val_loader = DataLoader(val_dataset, **eval_config)
    test_loader = DataLoader(test_dataset, **eval_config)

    return train_loader, val_loader, test_loader
