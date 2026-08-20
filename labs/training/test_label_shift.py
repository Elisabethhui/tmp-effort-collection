"""Tests for the explicit R4 causal loss Lab."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from labs.training.label_shift import (
    IGNORE_INDEX,
    causal_lm_loss,
    causal_lm_loss_sum_and_count,
    count_valid_shifted_targets,
    shift_causal_logits_and_labels,
)


class CausalLossTest(unittest.TestCase):
    def test_shift_drops_last_logit_and_first_label(self) -> None:
        logits = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
        labels = torch.tensor([[10, 11, 12, 13]])

        shifted_logits, shifted_labels = shift_causal_logits_and_labels(logits, labels)

        self.assertEqual(shifted_logits.shape, (1, 3, 3))
        self.assertEqual(shifted_labels.tolist(), [[11, 12, 13]])
        torch.testing.assert_close(shifted_logits, logits[:, :-1])

    def test_loss_matches_pytorch_on_shifted_targets_and_masks(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 4, 5, requires_grad=True)
        labels = torch.tensor(
            [[IGNORE_INDEX, 2, 3, 4], [IGNORE_INDEX, 1, IGNORE_INDEX, 0]]
        )

        actual = causal_lm_loss(logits, labels)
        loss_sum, valid_count = causal_lm_loss_sum_and_count(logits, labels)
        shifted_logits, shifted_labels = shift_causal_logits_and_labels(logits, labels)
        expected = F.cross_entropy(
            shifted_logits.reshape(-1, 5),
            shifted_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(loss_sum / valid_count, expected)
        actual.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertEqual(count_valid_shifted_targets(labels), 5)

    def test_empty_shifted_targets_are_rejected(self) -> None:
        logits = torch.zeros(1, 3, 4)
        labels = torch.full((1, 3), IGNORE_INDEX)

        with self.assertRaises(ValueError):
            causal_lm_loss(logits, labels)
        self.assertEqual(count_valid_shifted_targets(labels), 0)

    def test_bad_shapes_and_short_sequences_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            shift_causal_logits_and_labels(torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.long))
        with self.assertRaises(ValueError):
            shift_causal_logits_and_labels(
                torch.zeros(2, 4, 3), torch.zeros(2, 5, dtype=torch.long)
            )
        with self.assertRaises(ValueError):
            count_valid_shifted_targets(torch.zeros(2, 1, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
