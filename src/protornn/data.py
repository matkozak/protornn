from os import PathLike
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from protornn.tokenizer import ProteinTokenizer


class SequenceDataset(Dataset):
    """"""

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
    sample: float = 1.0,
    seed: int = 2137,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders from a FASTA file."""
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
    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)
    test_dataset = SequenceDataset(test_sequences)
    loader_kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        collate_fn=create_collate_fn(tokenizer, max_length=max_length),
        pin_memory=True,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
