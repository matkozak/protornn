import logging
from dataclasses import asdict
from os import PathLike
from pathlib import Path

import click
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from .data import create_dataloaders
from .model import ProtoRNN
from .tokenizer import ProteinTokenizer
from .train import train_model
from .utils import get_device, get_dtype, get_run_name, setup_logging

logger = logging.getLogger(__name__)


def run_experiment(
    data_path: str | PathLike,
    run_dir: str | PathLike,
    *,
    # Data params
    batch_size: int = 32,
    sample_size: float = 1.0,
    pack_sequences: bool = True,
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
        pack_sequences=pack_sequences,
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


@click.group()
def cli():
    pass


@cli.command("train")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--batch-size", default=32, help="Training batch size")
@click.option("--sample-size", default=1.0, help="Fraction of dataset to use")
@click.option("--embed-dim", default=64, help="Embedding dimension")
@click.option("--hidden-dim", default=1024, help="Hidden layer dimension")
@click.option("--num-layers", default=3, help="Number of RNN layers")
@click.option("--dropout", default=0.1, help="Dropout probability")
@click.option("--dtype", default="float32", help="Model dtype")
@click.option("--learning-rate", default=1e-3, help="Learning rate")
@click.option("--max-epochs", default=100, help="Maximum training epochs")
@click.option(
    "--tie-weights/--no-tie-weights",
    default=False,
    help="Tie encoder-decoder weights",
)
@click.option(
    "--pack-sequences/--no-pack-sequences",
    default=True,
    help="Batch by sequence length to reduce padding",
)
def train(data_path: Path, **kwargs):
    run_dir = Path("runs") / get_run_name()
    setup_logging(run_dir.with_suffix(".log"))
    run_experiment(data_path, run_dir, **kwargs)


@cli.command()
@click.argument("checkpoint", type=click.Path(exists=True, path_type=Path))
@click.argument("sequence", type=str)
def predict(checkpoint: Path, sequence: str):
    # TODO: Implement inference
    pass


if __name__ == "__main__":
    cli()
