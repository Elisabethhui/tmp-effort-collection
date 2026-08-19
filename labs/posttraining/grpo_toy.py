"""Small, explicit GRPO loss components for interview practice.

This is a mathematical toy, not a production trainer. It deliberately keeps
sequence log-probabilities as inputs so the policy-gradient data flow is easy
to inspect and test.
"""

from __future__ import annotations

import torch
from torch import Tensor


def group_relative_advantage(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize rewards within each prompt's group.

    Args:
        rewards: Tensor shaped [B, G], where G candidates share one prompt.
    """

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [B, G]")
    if rewards.shape[1] < 2:
        raise ValueError("each group needs at least two candidates")
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / std.clamp_min(eps)


def clipped_policy_objective(
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> tuple[Tensor, Tensor]:
    """Return the negative clipped surrogate loss and unclipped ratios.

    All inputs are sequence-level tensors shaped [B, G].
    """

    if new_logprobs.shape != old_logprobs.shape or new_logprobs.shape != advantages.shape:
        raise ValueError("new_logprobs, old_logprobs and advantages must have the same shape")
    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be positive")

    ratios = torch.exp(new_logprobs - old_logprobs)
    clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratios * advantages, clipped * advantages)
    return -surrogate.mean(), ratios


def reference_kl(new_logprobs: Tensor, reference_logprobs: Tensor) -> Tensor:
    """Compute the common non-negative log-probability KL approximation."""

    if new_logprobs.shape != reference_logprobs.shape:
        raise ValueError("new_logprobs and reference_logprobs must have the same shape")
    log_ratio = reference_logprobs - new_logprobs
    return (torch.exp(log_ratio) - log_ratio - 1.0).mean()


def grpo_loss(
    rewards: Tensor,
    new_logprobs: Tensor,
    old_logprobs: Tensor,
    reference_logprobs: Tensor,
    *,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.0,
) -> dict[str, Tensor]:
    """Compute toy GRPO statistics and a loss suitable for backpropagation."""

    advantages = group_relative_advantage(rewards)
    policy_loss, ratios = clipped_policy_objective(
        new_logprobs,
        old_logprobs,
        advantages,
        clip_epsilon=clip_epsilon,
    )
    kl = reference_kl(new_logprobs, reference_logprobs)
    return {
        "loss": policy_loss + kl_beta * kl,
        "policy_loss": policy_loss,
        "kl": kl,
        "advantages": advantages,
        "ratios": ratios,
    }
