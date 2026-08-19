"""Tests for the minimal SFT loss lab."""

from __future__ import annotations

import unittest

import torch

from labs.posttraining.sft_loss import (
    IGNORE_INDEX,
    build_answer_only_labels,
    causal_lm_loss,
)


class SFTLossTest(unittest.TestCase):
    def test_masks_prompt_and_padding(self) -> None:
        input_ids = torch.tensor([[10, 11, 20, 21, 0], [30, 40, 41, 0, 0]])
        prompt_lengths = torch.tensor([2, 1])
        attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])

        labels = build_answer_only_labels(input_ids, prompt_lengths, attention_mask)
        expected = torch.tensor(
            [[IGNORE_INDEX, IGNORE_INDEX, 20, 21, IGNORE_INDEX],
             [IGNORE_INDEX, 40, 41, IGNORE_INDEX, IGNORE_INDEX]]
        )
        torch.testing.assert_close(labels, expected)

    def test_loss_shifts_targets_and_backpropagates(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 4, 7, requires_grad=True)
        labels = torch.tensor([[IGNORE_INDEX, 2, 3, 4], [IGNORE_INDEX, 1, 5, IGNORE_INDEX]])

        loss = causal_lm_loss(logits, labels)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_rejects_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            causal_lm_loss(torch.randn(2, 1, 4), torch.ones(2, 1, dtype=torch.long))
        with self.assertRaises(ValueError):
            build_answer_only_labels(
                torch.ones(2, 4, dtype=torch.long), torch.tensor([0])
            )


if __name__ == "__main__":
    unittest.main()
