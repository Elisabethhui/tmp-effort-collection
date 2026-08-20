"""A tiny CPU-first decoder-only language model for R4.

The model deliberately reuses the explicit Transformer block from the R3
attention lab. It exposes the same ``[B, T] -> [B, T, V]`` path used by a
decoder-only language model, while keeping the implementation small enough to
inspect during an interview.

References:
- https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from labs.attention.transformer_block import RMSNorm, TransformerBlock


class TinyDecoderLM(nn.Module):
    """Small decoder-only LM with optional tied token embedding/output head."""

    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        num_kv_heads: int | None = None,
        ffn_dim: int | None = None,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        values = {
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError("vocab_size, max_seq_len, dimensions, and layer count must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            # Weight tying is a shared Parameter, not a detached copy.
            self.lm_head.weight = self.token_embedding.weight
        self.tie_weights = tie_weights

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        """Return next-token logits with shape ``[B, T, vocab_size]``.

        ``attention_mask`` uses the common convention 1/True for real tokens
        and 0/False for padding. The underlying reference block expects the
        inverse key-padding mask, so the conversion is explicit here.
        """

        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be an integer tensor with shape [B, T]")
        batch, seq_len = input_ids.shape
        if seq_len == 0 or seq_len > self.max_seq_len:
            raise ValueError("sequence length must be in [1, max_seq_len]")
        if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= self.vocab_size):
            raise ValueError("input_ids contain a token outside the vocabulary")

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
        elif position_ids.shape not in {(seq_len,), (batch, seq_len)}:
            raise ValueError("position_ids must have shape [T] or [B, T]")
        position_ids = position_ids.to(device=input_ids.device, dtype=torch.long)
        if position_ids.min() < 0 or position_ids.max() >= self.max_seq_len:
            raise ValueError("position_ids exceed max_seq_len")

        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have shape [B, T]")
            if attention_mask.dtype not in (torch.bool, torch.int32, torch.int64):
                raise TypeError("attention_mask must be bool or integer")
            real_tokens = attention_mask.to(dtype=torch.bool)
            if (~real_tokens).all(dim=-1).any():
                raise ValueError("each sequence must keep at least one real token")
            key_padding_mask = ~real_tokens
        else:
            key_padding_mask = None

        hidden = self.token_embedding(input_ids)
        position = self.position_embedding(position_ids)
        if position.ndim == 2:
            position = position.unsqueeze(0)
        hidden = hidden + position
        for block in self.blocks:
            hidden = block(
                hidden,
                causal=True,
                key_padding_mask=key_padding_mask,
                position_ids=position_ids,
            )
        return self.lm_head(self.final_norm(hidden))
