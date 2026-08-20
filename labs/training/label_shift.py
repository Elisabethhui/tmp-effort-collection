"""Explicit causal language-model label shifting and loss.

The implementation keeps the alignment visible:

    logits[:, :-1] predicts labels[:, 1:]

This is a correctness reference for tiny CPU experiments, not a fused kernel.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


IGNORE_INDEX = -100


def shift_causal_logits_and_labels(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return aligned next-token logits and targets.

    Args:
        logits: Unnormalized scores with shape ``[B, T, V]``.
        labels: Token ids with shape ``[B, T]``. ``IGNORE_INDEX`` values are
            preserved and can be used to mask prompt or padding targets.

    Returns:
        ``(shifted_logits, shifted_labels)`` with shapes ``[B, T-1, V]`` and
        ``[B, T-1]``.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if labels.ndim != 2 or labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [B, T] matching logits")
    if logits.shape[1] < 2:
        raise ValueError("sequence length must be at least 2")
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def causal_lm_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Compute masked next-token cross entropy from unnormalized logits.

    The denominator is the number of non-ignored shifted targets. An empty
    target set raises a clear error instead of returning a misleading NaN.
    """

    loss_sum, valid_count = causal_lm_loss_sum_and_count(
        logits,
        labels,
        ignore_index=ignore_index,
    )
    return (loss_sum / valid_count).to(dtype=logits.dtype)


def causal_lm_loss_sum_and_count(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[Tensor, int]:
    """Return the summed shifted loss and valid-token count.

    Train loops use this form to make gradient accumulation exact when
    micro-batches contain different numbers of non-ignored targets.
    """

    shifted_logits, shifted_labels = shift_causal_logits_and_labels(logits, labels)
    flat_logits = shifted_logits.float().reshape(-1, shifted_logits.shape[-1])
    flat_labels = shifted_labels.reshape(-1).long()
    valid = flat_labels != ignore_index
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ValueError("at least one non-ignored target is required")
    losses = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    return losses[valid].sum().to(dtype=logits.dtype), valid_count


def count_valid_shifted_targets(labels: Tensor, *, ignore_index: int = IGNORE_INDEX) -> int:
    """Count non-ignored targets that actually participate in causal loss."""

    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must have shape [B, T] with T >= 2")
    return int((labels[:, 1:] != ignore_index).sum().item())
