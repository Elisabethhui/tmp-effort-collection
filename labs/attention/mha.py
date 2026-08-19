"""Interview-sized, explicit Multi-Head Attention implementation.

The reference semantics are PyTorch's scaled_dot_product_attention:
https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ManualMultiHeadAttention(nn.Module):
    """Self-attention with explicit projections, shapes, and masks.

    Args:
        d_model: Hidden dimension D.
        num_heads: Number of attention heads H. D must be divisible by H.
        dropout: Attention-probability dropout used only in training mode.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        # transpose makes the attention dimensions [B, H, T, Dh].
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        *,
        causal: bool = False,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, T, D], got {tuple(x.shape)}")
        batch, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"expected D={self.d_model}, got D={d_model}")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, seq_len):
                raise ValueError(
                    "key_padding_mask must have shape [B, T] with True meaning padding"
                )
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must be a bool tensor")
            if key_padding_mask.all(dim=-1).any():
                raise ValueError("each sequence must keep at least one unmasked key")

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal:
            causal_mask = torch.ones(
                seq_len, seq_len, device=x.device, dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            # Mask key positions only; query padding is handled by the caller.
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min
            )

        # Compute probabilities in fp32 for a more stable softmax, then restore dtype.
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(context)
