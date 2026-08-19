"""Tests for dependency-free retrieval metrics."""

from __future__ import annotations

import unittest

from labs.rag.metrics import ndcg_at_k, recall_at_k, reciprocal_rank, rrf_fuse


class RetrievalMetricsTest(unittest.TestCase):
    def test_recall_and_reciprocal_rank(self) -> None:
        relevant = {"a", "c"}
        retrieved = ["x", "c", "a"]
        self.assertEqual(recall_at_k(relevant, retrieved, 1), 0.0)
        self.assertEqual(recall_at_k(relevant, retrieved, 3), 1.0)
        self.assertAlmostEqual(reciprocal_rank(relevant, retrieved), 0.5)

    def test_ndcg_rewards_high_relevance_early(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([3.0, 0.0, 1.0], 3), 0.9639, places=3)
        self.assertEqual(ndcg_at_k([0.0, 0.0], 2), 0.0)

    def test_rrf_fusion_is_deterministic_and_deduplicates(self) -> None:
        fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], rank_constant=1)
        self.assertEqual(fused[0], "b")
        self.assertEqual(set(fused), {"a", "b", "c", "d"})

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k({"a"}, ["a"], 0)
        with self.assertRaises(ValueError):
            ndcg_at_k([1.0], 0)


if __name__ == "__main__":
    unittest.main()
