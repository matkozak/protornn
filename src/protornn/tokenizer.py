from typing import Sequence

PAD_TOK = "_"
UNK_TOK = "?"
BOS_TOK = "^"
EOS_TOK = "*"
MASK_TOK = "#"


class ProteinTokenizer:
    def __init__(self):
        self.special_tokens = (PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK, MASK_TOK)
        self.aa_tokens = "ACDEFGHIKLMNPQRSTVWY"
        self.vocab = (*self.special_tokens, *self.aa_tokens)

        self.tok_to_idx = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.idx_to_tok = {idx: tok for idx, tok in enumerate(self.vocab)}

    def encode(
        self,
        sequence: str,
        add_bos: bool = False,
        add_eos: bool = False,
        pad_to: int | None = None,
    ) -> list[int]:
        tokens = list(sequence.upper())

        if add_bos:
            tokens.insert(0, BOS_TOK)
        if add_eos:
            tokens.append(EOS_TOK)

        encoded = [self.tok_to_idx.get(tok, self.tok_to_idx[UNK_TOK]) for tok in tokens]

        if pad_to is not None:
            encoded = encoded[:pad_to]  # truncate if pad_to < len(sequence)
            padding = [self.tok_to_idx[PAD_TOK]] * (pad_to - len(encoded))
            encoded.extend(padding)

        return encoded

    def batch_encode(
        self, sequences: Sequence[str], *args, **kwargs
    ) -> list[list[int]]:
        if isinstance(sequences, str):
            sequences = [sequences]
        encoded = [self.encode(seq, *args, **kwargs) for seq in sequences]
        return encoded

    def decode(self, indices: Sequence[int], strip_special: bool = True) -> str:
        if strip_special:
            special = {self.tok_to_idx[tok] for tok in self.special_tokens}
            indices = [idx for idx in indices if idx not in special]
        decoded = "".join(self.idx_to_tok.get(idx, UNK_TOK) for idx in indices)
        return decoded

    @property
    def pad_idx(self):
        return self.tok_to_idx[PAD_TOK]
