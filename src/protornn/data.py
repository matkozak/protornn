from os import PathLike

import torch
from torch.utils.data import Dataset

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
        dt = torch.uint8

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
