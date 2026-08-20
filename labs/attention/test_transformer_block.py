"""Tests for the CPU-first Transformer block."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed; run after environment setup")
class TransformerBlockTest(unittest.TestCase):
    def test_rope_position_zero_is_identity(self) -> None:
        from labs.attention.transformer_block import apply_rope

        q = torch.randn(2, 3, 1, 8)
        k = torch.randn(2, 2, 1, 8)
        rotated_q, rotated_k = apply_rope(q, k, position_ids=torch.zeros(1, dtype=torch.long))
        torch.testing.assert_close(rotated_q, q)
        torch.testing.assert_close(rotated_k, k)

    def test_gqa_block_shape_and_backward(self) -> None:
        from labs.attention.transformer_block import TransformerBlock

        torch.manual_seed(0)
        block = TransformerBlock(d_model=16, num_heads=4, num_kv_heads=2, ffn_dim=32)
        x = torch.randn(2, 5, 16, requires_grad=True)
        output = block(x)
        self.assertEqual(output.shape, x.shape)
        output.square().mean().backward()
        self.assertIsNotNone(block.q_proj.weight.grad)

    def test_causal_future_change_does_not_change_past(self) -> None:
        from labs.attention.transformer_block import TransformerBlock

        torch.manual_seed(1)
        block = TransformerBlock(d_model=8, num_heads=2, ffn_dim=16).eval()
        x = torch.randn(1, 4, 8)
        changed = x.clone()
        changed[:, 3] += 100.0
        first = block(x)
        second = block(changed)
        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=1e-5, atol=1e-5)

    def test_rejects_invalid_grouped_heads_and_masks(self) -> None:
        from labs.attention.transformer_block import TransformerBlock

        with self.assertRaises(ValueError):
            TransformerBlock(d_model=12, num_heads=3, num_kv_heads=2)
        block = TransformerBlock(d_model=8, num_heads=2)
        with self.assertRaises(ValueError):
            block(torch.randn(1, 3, 8), key_padding_mask=torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()
