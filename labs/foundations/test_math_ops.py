"""Tests for numerically explicit math primitives."""

from __future__ import annotations

import unittest

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError:
    torch = None
    F = None


@unittest.skipIf(torch is None, "PyTorch is not installed; run after environment setup")
class MathOpsTest(unittest.TestCase):
    def test_stable_softmax_handles_large_logits(self) -> None:
        from labs.foundations.math_ops import stable_softmax

        output = stable_softmax(torch.tensor([[10000.0, 10001.0, 9999.0]]))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output.sum(dim=-1), torch.ones(1), atol=1e-3, rtol=1e-3)

    def test_cross_entropy_matches_pytorch_and_masks(self) -> None:
        from labs.foundations.math_ops import cross_entropy_from_logits

        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
        targets = torch.tensor([2, -100])
        actual = cross_entropy_from_logits(logits, targets)
        expected = F.cross_entropy(logits, targets)
        torch.testing.assert_close(actual, expected)

    def test_kl_is_zero_for_identical_distributions(self) -> None:
        from labs.foundations.math_ops import kl_divergence_from_log_probs

        log_p = torch.log_softmax(torch.tensor([[1.0, 2.0, 4.0]]), dim=-1)
        actual = kl_divergence_from_log_probs(log_p, log_p)
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=1e-6, rtol=1e-6)

    def test_adam_update_changes_parameter_and_tracks_moments(self) -> None:
        from labs.foundations.math_ops import adam_update

        parameter = torch.tensor([1.0])
        gradient = torch.tensor([2.0])
        updated, first, second = adam_update(
            parameter,
            gradient,
            torch.zeros_like(parameter),
            torch.zeros_like(parameter),
            1,
        )
        self.assertLess(updated.item(), parameter.item())
        self.assertGreater(first.item(), 0.0)
        self.assertGreater(second.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
