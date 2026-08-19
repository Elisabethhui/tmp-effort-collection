"""Tests for the manual attention lab.

Run with:
    python -m unittest labs/attention/test_mha.py -v
"""

from __future__ import annotations

import unittest

try:
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError:  # Keep the repository test-discoverable before setup.
    torch = None
    F = None


@unittest.skipIf(torch is None, "PyTorch is not installed; run after environment setup")
class ManualMultiHeadAttentionTest(unittest.TestCase):
    def test_matches_pytorch_sdpa_for_causal_attention(self) -> None:
        from labs.attention.mha import ManualMultiHeadAttention

        torch.manual_seed(0)
        model = ManualMultiHeadAttention(d_model=16, num_heads=4).eval()
        x = torch.randn(2, 5, 16)

        actual = model(x, causal=True)
        q = model._split_heads(model.q_proj(x))
        k = model._split_heads(model.k_proj(x))
        v = model._split_heads(model.v_proj(x))
        expected_heads = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        expected = expected_heads.transpose(1, 2).contiguous().view(2, 5, 16)
        expected = model.out_proj(expected)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_padding_mask_blocks_padded_keys(self) -> None:
        from labs.attention.mha import ManualMultiHeadAttention

        torch.manual_seed(1)
        model = ManualMultiHeadAttention(d_model=8, num_heads=2).eval()
        x = torch.randn(1, 4, 8)
        padding = torch.tensor([[False, False, False, True]])
        changed = x.clone()
        changed[:, 3] = torch.randn(8) * 1000

        actual = model(x, key_padding_mask=padding)
        changed_output = model(changed, key_padding_mask=padding)
        torch.testing.assert_close(actual[:, :3], changed_output[:, :3], rtol=1e-5, atol=1e-5)

    def test_shape_and_invalid_inputs(self) -> None:
        from labs.attention.mha import ManualMultiHeadAttention

        model = ManualMultiHeadAttention(d_model=12, num_heads=3)
        output = model(torch.randn(2, 7, 12))
        self.assertEqual(output.shape, (2, 7, 12))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 7, 10))


if __name__ == "__main__":
    unittest.main()
