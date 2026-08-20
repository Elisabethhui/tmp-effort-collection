"""Tests for the R4 tiny decoder-only language model."""

from __future__ import annotations

import unittest

import torch

from labs.training.tiny_lm import TinyDecoderLM


class TinyDecoderLMTest(unittest.TestCase):
    def test_logits_shape_weight_tying_and_backward(self) -> None:
        torch.manual_seed(0)
        model = TinyDecoderLM(
            vocab_size=19,
            max_seq_len=8,
            d_model=16,
            num_heads=4,
            num_kv_heads=2,
            ffn_dim=32,
        )
        input_ids = torch.randint(0, 19, (2, 5))

        logits = model(input_ids)

        self.assertEqual(logits.shape, (2, 5, 19))
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        logits.square().mean().backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertGreater(float(model.token_embedding.weight.grad.abs().sum()), 0.0)

    def test_future_token_does_not_change_past_logits(self) -> None:
        torch.manual_seed(1)
        model = TinyDecoderLM(
            vocab_size=13,
            max_seq_len=8,
            d_model=16,
            num_heads=4,
            num_layers=2,
        ).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        changed = input_ids.clone()
        changed[:, 3] = 12

        first = model(input_ids)
        second = model(changed)

        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=1e-5, atol=1e-5)

    def test_attention_mask_blocks_padded_key(self) -> None:
        torch.manual_seed(2)
        model = TinyDecoderLM(
            vocab_size=11,
            max_seq_len=6,
            d_model=16,
            num_heads=4,
        ).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        changed = input_ids.clone()
        changed[:, 3] = 10
        mask = torch.tensor([[1, 1, 1, 0]])

        first = model(input_ids, attention_mask=mask)
        second = model(changed, attention_mask=mask)

        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=1e-5, atol=1e-5)

    def test_rejects_invalid_input_and_positions(self) -> None:
        model = TinyDecoderLM(vocab_size=8, max_seq_len=4, d_model=8, num_heads=2)
        with self.assertRaises(ValueError):
            model(torch.ones(1, 2, dtype=torch.float32))
        with self.assertRaises(ValueError):
            model(torch.tensor([[0, 8]]))
        with self.assertRaises(ValueError):
            model(torch.tensor([[0, 1]]), position_ids=torch.tensor([0, 4]))
        with self.assertRaises(ValueError):
            model(torch.tensor([[0, 1]]), attention_mask=torch.zeros(1, 2, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
