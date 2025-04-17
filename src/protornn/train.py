import logging
import os
from typing import TypedDict

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import ProtoRNN
from .stopper import EarlyStopper

logger = logging.getLogger(__name__)
Metrics = TypedDict(
    "Metrics",
    {
        "train/loss": float,
        "val/loss": float,
    },
)


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    """Train model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_size = 0

    pbar = tqdm(data_loader, leave=False, position=1, desc="Training batch")
    for batch in pbar:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        x, y = (t.to(device) for t in batch)
        logits: torch.Tensor = model(x)  # (batch_size, seq_len, vocab_size)
        loss = criterion(
            logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab_size)
            y.view(-1),  # (batch_size * seq_len)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        total_size += len(x)

    return total_loss / total_size


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_size = 0

    pbar = tqdm(data_loader, leave=False, position=1, desc="Validation batch")
    for batch in pbar:
        # forward + backward + optimize
        x, y = (t.to(device) for t in batch)
        logits: torch.Tensor = model(x)  # (batch_size, seq_len, vocab_size)
        loss = criterion(
            logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab_size)
            y.view(-1),  # (batch_size * seq_len)
        )
        total_loss += loss.item() * len(x)
        total_size += len(x)

    return total_loss / total_size


def train_model(
    model: ProtoRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    num_epochs: int,
    device: torch.device,
    early_stop: bool = True,
    checkpoint_path: str | os.PathLike | None = None,
):
    """"""
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    stopper = EarlyStopper()

    pbar = tqdm(range(num_epochs), desc="Epoch", leave=False, initial=1)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        stopper.step(val_loss)

        metrics: Metrics = {"train/loss": train_loss, "val/loss": val_loss}
        yield metrics

        if stopper.save and checkpoint_path is not None:
            torch.save(model.state_dict(), checkpoint_path)
        if stopper.stop and early_stop:
            break
