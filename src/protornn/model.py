import torch
from torch import nn


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        use_attention: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_rnn = nn.LayerNorm(embed_dim)
        self.ln_attn = nn.LayerNorm(embed_dim)
        self.ln_ffn = nn.LayerNorm(embed_dim)

        self.rnn = nn.LSTM(embed_dim, embed_dim, batch_first=True)

        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True
            )

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        normed_x = self.ln_rnn(x)
        rnn_out, _ = self.rnn(normed_x)
        x = x + self.dropout(rnn_out)  # Residual connection

        if self.use_attention:
            normed_x = self.ln_attn(x)
            if mask is None:
                seq_len = x.size(1)  # (batch_size, sequence_length, embed_dim)
                mask = torch.triu(
                    torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
                    diagonal=1,
                )
            # TODO: padding mask
            attn_out, _ = self.attention(normed_x, normed_x, normed_x, attn_mask=mask)
            x = x + self.dropout(attn_out)  # Residual connection

        normed_x = self.ln_ffn(x)
        ff_out = self.ff(normed_x)
        x = x + ff_out  # Residual connection

        return x


class ProtoRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    use_attention=(i == num_layers - 2),
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.decoder = nn.Linear(embed_dim, vocab_size)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        x = self.encoder(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
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
