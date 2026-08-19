"""Tests for the toy GRPO objective."""

from __future__ import annotations

import unittest

import torch

from labs.posttraining.grpo_toy import (
    clipped_policy_objective,
    group_relative_advantage,
    grpo_loss,
    reference_kl,
)


class ToyGRPOTest(unittest.TestCase):
    def test_group_advantage_is_centered_and_constant_groups_are_safe(self) -> None:
        rewards = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
        advantages = group_relative_advantage(rewards)
        torch.testing.assert_close(advantages.mean(dim=1), torch.zeros(2))
        self.assertTrue(torch.isfinite(advantages).all().item())
        torch.testing.assert_close(advantages[1], torch.zeros(3))

    def test_policy_ratio_clipping(self) -> None:
        old = torch.zeros(1, 3)
        new = torch.log(torch.tensor([[3.0, 0.5, 1.1]]))
        advantages = torch.tensor([[1.0, 1.0, 1.0]])
        _, ratios = clipped_policy_objective(new, old, advantages, clip_epsilon=0.2)
        torch.testing.assert_close(ratios, torch.tensor([[3.0, 0.5, 1.1]]))

    def test_loss_has_finite_gradient_and_kl_is_zero_for_identical_policies(self) -> None:
        torch.manual_seed(0)
        new = torch.randn(2, 4, requires_grad=True)
        old = torch.randn(2, 4)
        reference = new.detach().clone()
        rewards = torch.randn(2, 4)
        result = grpo_loss(rewards, new, old, reference, kl_beta=0.1)
        self.assertTrue(torch.isfinite(result["loss"]).item())
        result["loss"].backward()
        self.assertTrue(torch.isfinite(new.grad).all().item())
        torch.testing.assert_close(reference_kl(reference, reference), torch.tensor(0.0))

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            group_relative_advantage(torch.ones(2, 1))
        with self.assertRaises(ValueError):
            clipped_policy_objective(torch.ones(2, 2), torch.ones(2, 3), torch.ones(2, 2))


if __name__ == "__main__":
    unittest.main()
