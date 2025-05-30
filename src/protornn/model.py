import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        """Standard multi-head self-attention implementation"""
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("Attention size must be divisible by number of heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = dropout
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L = x.size()[:2]
        E, H_n = self.embed_dim, self.num_heads
        H_d = E // H_n
        q, k, v = self.qkv_proj(x).split(self.embed_dim, dim=-1)
        q = q.view(B, L, H_n, H_d).transpose(1, 2)  # B, H_n, L, H_d
        k = k.view(B, L, H_n, H_d).transpose(1, 2)
        v = v.view(B, L, H_n, H_d).transpose(1, 2)

        attn_dropout = self.attn_dropout if self.training else 0
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=attn_dropout
        )
        # Reshape and project output
        attn = attn.transpose(1, 2).view(B, L, E).contiguous()
        attn = self.out_dropout(self.out_proj(attn))
        return attn


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
            self.attention = SelfAttention(embed_dim=embed_dim, dropout=dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        normed_x = self.ln_rnn(x)
        rnn_out, _ = self.rnn(normed_x)
        x = x + self.dropout(rnn_out)  # Residual connection

        if self.use_attention:
            normed_x = self.ln_attn(x)
            # TODO: evaluate q, k, v norm
            attn_out = self.attention(normed_x, attn_mask)
            x = x + 0  # this stops MPS from having a fit, pls dont ask me why
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
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
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
        causal_mask = torch.tril(
            torch.ones((1, 1, seq_len, seq_len), device=x.device, dtype=torch.bool),
            diagonal=0,
        )  # broadcastable to (batch_size, num_heads, seq_len, seq_len)
        padding_mask = x != self.pad_idx  # (batch_size, seq_len)
        attn_mask = causal_mask & padding_mask.unsqueeze(1).unsqueeze(1)
        x = self.encoder(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attn_mask)
            x = x * padding_mask.unsqueeze(-1)
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
