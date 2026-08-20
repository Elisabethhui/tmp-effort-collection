"""Numerically explicit math primitives for interview practice."""

from __future__ import annotations

import torch
from torch import Tensor


def stable_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Compute softmax through log-sum-exp in a safer accumulator dtype."""

    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension")
    work = logits.float()
    return (work - torch.logsumexp(work, dim=dim, keepdim=True)).exp().to(logits.dtype)


def cross_entropy_from_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """Compute class cross-entropy without materializing softmax probabilities."""

    if logits.ndim < 2:
        raise ValueError("logits must have class dimension")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits without the class dimension")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean, or sum")

    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    safe_targets = flat_targets.masked_fill(~valid, 0).long()
    log_probs = flat_logits - torch.logsumexp(flat_logits, dim=-1, keepdim=True)
    losses = -log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    losses = losses.masked_fill(~valid, 0.0)
    if reduction == "none":
        return losses.reshape(targets.shape).to(logits.dtype)
    if reduction == "sum":
        return losses.sum().to(logits.dtype)
    denominator = valid.sum().clamp_min(1).to(losses.dtype)
    return (losses.sum() / denominator).to(logits.dtype)


def kl_divergence_from_log_probs(log_p: Tensor, log_q: Tensor, dim: int = -1) -> Tensor:
    """Compute KL(P||Q) from normalized log-probabilities."""

    if log_p.shape != log_q.shape:
        raise ValueError("log_p and log_q must have the same shape")
    p = log_p.float().exp()
    return (p * (log_p.float() - log_q.float())).sum(dim=dim)


def adam_update(
    parameter: Tensor,
    gradient: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    step: int,
    *,
    learning_rate: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return one decoupled-weight-decay AdamW-style update."""

    if step <= 0:
        raise ValueError("step must be positive")
    if parameter.shape != gradient.shape or parameter.shape != first_moment.shape or parameter.shape != second_moment.shape:
        raise ValueError("parameter, gradient, and moments must have identical shapes")
    beta1, beta2 = betas
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0 or learning_rate < 0.0 or eps <= 0.0:
        raise ValueError("invalid Adam hyperparameters")

    m = beta1 * first_moment + (1.0 - beta1) * gradient
    v = beta2 * second_moment + (1.0 - beta2) * gradient.square()
    m_hat = m / (1.0 - beta1**step)
    v_hat = v / (1.0 - beta2**step)
    updated = parameter * (1.0 - learning_rate * weight_decay) - learning_rate * m_hat / (v_hat.sqrt() + eps)
    return updated, m, v
