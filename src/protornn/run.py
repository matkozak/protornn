from datetime import datetime

import torch

from protornn.data import create_dataloaders
from protornn.model import ProtoRNN
from protornn.tokenizer import ProteinTokenizer
from protornn.train import train_model
from protornn.utils import get_device, setup_logging


def main():
    device = get_device()
    tokenizer = ProteinTokenizer()
    train_data, val_data, test_data = create_dataloaders(
        "data/uniprot_sprot.fasta", tokenizer, batch_size=32, sample=0.05
    )
    model = ProtoRNN(len(tokenizer.vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = train_model(model, train_data, val_data, optimizer, 100, device)
    for metrics in history:
        pass


if __name__ == "__main__":
    setup_logging(
        f"checkpoints/{datetime.now().isoformat(timespec='minutes')}_train.log"
    )
    main()
