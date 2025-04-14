import torch
from torch import nn


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.ff(x)


class ProtoRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.decoder(x)

        return x


if __name__ == "__main__":
    # Basic test to verify model functionality
    from protornn.tokenizer import ProteinTokenizer

    # Create a tokenizer
    tokenizer = ProteinTokenizer()
    vocab_size = len(tokenizer.vocab)

    # Initialize model
    model = ProtoRNN(vocab_size=vocab_size, embed_dim=64)

    # Create a dummy batch: [batch_size, sequence_length]
    batch_size = 4
    seq_length = 10
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Forward pass
    output = model(dummy_input)

    # Verify shapes
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_length, vocab_size)

    print("Basic test passed!")

    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
