"""Minimal answer-only causal language-modeling loss."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


IGNORE_INDEX = -100


def build_answer_only_labels(
    input_ids: Tensor,
    prompt_lengths: Tensor,
    attention_mask: Tensor | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Mask prompt and padding positions in a copy of ``input_ids``.

    ``prompt_lengths[b]`` is the number of leading tokens that belong to the
    system/user prompt for sample ``b``. ``attention_mask`` uses 1 for real
    tokens and 0 for padding.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    batch, seq_len = input_ids.shape
    if prompt_lengths.shape != (batch,):
        raise ValueError("prompt_lengths must have shape [B]")
    if prompt_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("prompt_lengths must be an integer tensor")
    if ((prompt_lengths < 0) | (prompt_lengths > seq_len)).any():
        raise ValueError("prompt lengths must be within [0, T]")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have shape [B, T]")
        if attention_mask.dtype not in (torch.bool, torch.int32, torch.int64):
            raise TypeError("attention_mask must be bool or integer")

    labels = input_ids.clone()
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    labels[positions < prompt_lengths.unsqueeze(1)] = ignore_index
    if attention_mask is not None:
        labels[attention_mask == 0] = ignore_index
    return labels


def causal_lm_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Compute next-token cross entropy for logits shaped ``[B, T, V]``."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [B, T]")
    if logits.shape[1] < 2:
        raise ValueError("sequence length must be at least 2")

    # Token t predicts token t+1; ignore_index removes prompt/pad targets.
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )
