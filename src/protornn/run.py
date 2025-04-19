from dataclasses import asdict
from datetime import datetime

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from protornn.data import create_dataloaders
from protornn.model import ProtoRNN
from protornn.tokenizer import ProteinTokenizer
from protornn.train import train_model
from protornn.utils import get_device, setup_logging


def main():
    device = get_device()
    tokenizer = ProteinTokenizer()
    writer = SummaryWriter()
    train_data, val_data, test_data = create_dataloaders(
        "data/uniprot_sprot.fasta", tokenizer, batch_size=32, sample=0.01
    )
    model = ProtoRNN(len(tokenizer.vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = train_model(model, train_data, val_data, optimizer, 100, device)
    for epoch, metrics in enumerate(history, 1):
        writer.add_scalars("Epoch metrics", asdict(metrics), epoch)
        writer.flush()
    writer.close()


if __name__ == "__main__":
    setup_logging(f"runs/{datetime.now().isoformat(timespec='minutes')}_train.log")
    main()
