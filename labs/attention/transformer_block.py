"""CPU-first decoder block for Transformer interview practice.

The implementation intentionally exposes the shapes used by GPT/Llama-style
blocks. It is a correctness reference, not a fused GPU kernel.

References:
- https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
- https://huggingface.co/docs/transformers/internal/rope_utils
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean subtraction."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0 or eps <= 0:
            raise ValueError("d_model and eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


def apply_rope(
    q: Tensor,
    k: Tensor,
    *,
    position_ids: Tensor | None = None,
    base: float = 10_000.0,
) -> tuple[Tensor, Tensor]:
    """Rotate Q/K pairs with RoPE; accepts [B, H, T, Dh] tensors."""

    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("q and k must have compatible shapes [B, H, T, Dh]")
    batch, _, seq_len, head_dim = q.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if base <= 1.0:
        raise ValueError("RoPE base must be greater than one")

    if position_ids is None:
        positions = torch.arange(seq_len, device=q.device)
    else:
        if position_ids.shape not in {(seq_len,), (batch, seq_len)}:
            raise ValueError("position_ids must have shape [T] or [B, T]")
        positions = position_ids.to(device=q.device)

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    angles = positions.float().unsqueeze(-1) * inv_freq
    cos = angles.cos()
    sin = angles.sin()
    if positions.ndim == 1:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]

    def rotate(x: Tensor) -> Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos.to(dtype=x.dtype) - odd * sin.to(dtype=x.dtype)
        rotated_odd = even * sin.to(dtype=x.dtype) + odd * cos.to(dtype=x.dtype)
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


def repeat_kv(hidden_states: Tensor, num_query_heads: int) -> Tensor:
    """Expand GQA/MQA K/V heads to the query-head count without copying first."""

    if hidden_states.ndim != 4:
        raise ValueError("hidden_states must have shape [B, Hkv, T, Dh]")
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if num_query_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be a positive multiple of num_kv_heads")
    repeats = num_query_heads // num_kv_heads
    if repeats == 1:
        return hidden_states
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, repeats, seq_len, head_dim)
        .reshape(batch, num_query_heads, seq_len, head_dim)
    )


class TransformerBlock(nn.Module):
    """A pre-norm causal decoder block with optional GQA and RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        num_kv_heads: int | None = None,
        ffn_dim: int | None = None,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_kv_heads <= 0 or num_heads % self.num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_heads")
        self.head_dim = d_model // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_base = rope_base

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, kv_dim)
        self.v_proj = nn.Linear(d_model, kv_dim)
        self.o_proj = nn.Linear(d_model, d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn_dim = ffn_dim or 4 * d_model
        self.gate_up_proj = nn.Linear(d_model, 2 * self.ffn_dim)
        self.down_proj = nn.Linear(self.ffn_dim, d_model)

    def _attention(self, x: Tensor, *, causal: bool, key_padding_mask: Tensor | None, position_ids: Tensor | None) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, position_ids=position_ids, base=self.rope_base)
        k = repeat_kv(k, self.num_heads)
        v = repeat_kv(v, self.num_heads)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal:
            causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, seq_len) or key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be bool with shape [B, T]")
            if key_padding_mask.all(dim=-1).any():
                raise ValueError("each sequence must keep at least one unmasked key")
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probabilities, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(context)

    def forward(
        self,
        x: Tensor,
        *,
        causal: bool = True,
        key_padding_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")
        attended = x + self._attention(
            self.norm1(x),
            causal=causal,
            key_padding_mask=key_padding_mask,
            position_ids=position_ids,
        )
        gate, value = self.gate_up_proj(self.norm2(attended)).chunk(2, dim=-1)
        feed_forward = self.down_proj(F.silu(gate) * value)
        return attended + feed_forward
