"""Tests for KV-cache memory calculations."""

from __future__ import annotations

import unittest

from labs.inference.memory import bytes_to_gib, kv_cache_bytes


class KVCacheMemoryTest(unittest.TestCase):
    def test_counts_both_k_and_v(self) -> None:
        self.assertEqual(
            kv_cache_bytes(
                batch_size=1,
                sequence_length=10,
                num_layers=2,
                num_kv_heads=4,
                head_dim=8,
                bytes_per_element=2,
            ),
            2560,
        )

    def test_gqa_reduces_memory_when_kv_heads_drop(self) -> None:
        mha = kv_cache_bytes(
            batch_size=2, sequence_length=1024, num_layers=8, num_kv_heads=16, head_dim=64, bytes_per_element=2
        )
        gqa = kv_cache_bytes(
            batch_size=2, sequence_length=1024, num_layers=8, num_kv_heads=4, head_dim=64, bytes_per_element=2
        )
        self.assertEqual(mha / gqa, 4.0)

    def test_invalid_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            kv_cache_bytes(
                batch_size=0,
                sequence_length=1,
                num_layers=1,
                num_kv_heads=1,
                head_dim=1,
                bytes_per_element=2,
            )
        self.assertAlmostEqual(bytes_to_gib(1024**3), 1.0)


if __name__ == "__main__":
    unittest.main()
