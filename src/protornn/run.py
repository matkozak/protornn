import logging
from dataclasses import asdict
from os import PathLike
from pathlib import Path

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from protornn.data import create_dataloaders
from protornn.model import ProtoRNN
from protornn.tokenizer import ProteinTokenizer
from protornn.train import train_model
from protornn.utils import get_device, get_dtype, get_run_name, setup_logging

logger = logging.getLogger(__name__)


def run_experiment(
    data_path: str | PathLike,
    run_dir: str | PathLike,
    *,
    # Data params
    batch_size: int = 32,
    sample_size: float = 1.0,
    # Model params
    embed_dim: int = 64,
    hidden_dim: int = 1024,
    num_layers: int = 3,
    dropout: float = 0.1,
    tie_weights: bool = False,
    dtype: str = "float32",
    # Training params
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
) -> None:
    device = get_device()
    tokenizer = ProteinTokenizer()
    writer = SummaryWriter(run_dir)

    train_data, val_data, test_data = create_dataloaders(
        data_path,
        tokenizer,
        batch_size=batch_size,
        sample=sample_size,
    )

    model = ProtoRNN(
        len(tokenizer.vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        tie_weights=tie_weights,
    ).to(device, get_dtype(dtype))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logger.info("Run: %s", run_dir)
    history = train_model(model, train_data, val_data, optimizer, max_epochs, device)
    for epoch, metrics in enumerate(history, 1):
        writer.add_scalars("Epoch metrics", asdict(metrics), epoch)
        writer.flush()
    writer.close()


if __name__ == "__main__":
    run_dir = Path("runs") / get_run_name()
    setup_logging(run_dir.with_suffix(".log"))
    run_experiment("data/uniprot_sprot.fasta", run_dir)
