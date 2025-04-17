from os import PathLike
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from protornn.tokenizer import ProteinTokenizer


class ProteinSequenceDataset(Dataset):
    """"""

    def __init__(
        self,
        sequences: list[str],
        tokenizer: ProteinTokenizer,
        max_length: int | None = None,
    ) -> None:
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length or max(map(len, sequences), default=0)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        dt = torch.long

        # Tokenize with BOS and EOS tokens
        input = self.tokenizer.encode(
            sequence,
            add_bos=True,
            add_eos=False,
            pad_to=self.max_length + 1,  # +1 for BOS
        )

        # Target is the sequence shifted by one (predict next token)
        target = self.tokenizer.encode(
            sequence,
            add_bos=False,
            add_eos=True,
            pad_to=self.max_length + 1,  # +1 for EOS
        )
        return torch.tensor(input, dtype=dt), torch.tensor(target, dtype=dt)


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
    train_dataset = ProteinSequenceDataset(train_sequences, tokenizer, max_length)
    val_dataset = ProteinSequenceDataset(val_sequences, tokenizer, max_length)
    test_dataset = ProteinSequenceDataset(test_sequences, tokenizer, max_length)

    loader_kwargs: dict[str, Any] = dict(batch_size=batch_size, pin_memory=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
