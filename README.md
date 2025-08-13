# ProtoRNN

An RNN protein language model built from scratch.
RNN backbone with optional attention, trained on SwissProt sequences.

## What this is

A weekend project to implement sequence modeling for proteins without using pretrained models.
The architecture combines LSTM layers with optional multi-head attention,
similar to early transformer experiments but keeping the RNN backbone for sequential processing.

Built this to understand what's actually happening under the hood in protein language models.

## Architecture

- Embedding layer with optional weight tying
- Stack of LSTM + residual blocks
- Optional attention layer (only on penultimate block by default)
- LayerNorm + feedforward network in each block
- Trained with teacher forcing on next-token prediction

Nothing groundbreaking, but it works and trains reasonably fast.

## Why RNN in 2025?

I wanted to build something from scratch and thought that
[SHA-RNN](https://arxiv.org/abs/1911.11423) was a fun paper.

Moreover:
1. Proteins are actually sequential.
1. I have found that a few LSTM layers can do just as well as a fine-tuned SOTA transformer
    on many basic discriminative tasks.

So this is an experiment to see if one can train a token-efficient base model
for property tasks using a laptop and some free cloud credits.

# Setup

## Code structure

```
model.py - Core architecture
data.py - Sequence datasets with length-based batching
tokenizer.py - Simple amino acid tokenization
train.py - Training loop with early stopping
utils.py - Device selection, logging, etc.
```

## Installation

```bash
git clone https://github.com/matkozak/protornn.git
cd protornn

# highly recommended, uv single-handedly solves all Python packaging woes
uv sync

# or just pip it
# python -m venv .venv
# pip install -e ".[train]"
```

## Training

```bash
# download SwissProt
bash scripts/get_data.sh

# use the CLI interface
protornn train data/uniprot_sprot.fasta \
    --batch-size 32 \
    --embed-dim 128 \
    --hidden-dim 2048 \
    --num-layers 4
```

Checkpoints and tensorboard logs go to runs/.
Takes a few hours on a decent GPU for meaningful results.


# Status
This is a side project for learning, not production code.
Goals include training a faithful SHA-RNN replication with clean code,
potentially moving on to Mamba and other modern improvements on LSTM.
