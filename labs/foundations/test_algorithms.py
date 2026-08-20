"""Tests for CPU-first interview data structures."""

from __future__ import annotations

import unittest

from labs.foundations.algorithms import LRUCache, bfs_distance, top_k


class AlgorithmsTest(unittest.TestCase):
    def test_top_k_and_edges(self) -> None:
        self.assertEqual(top_k([4, 1, 7, 3, 7], 3), [7, 7, 4])
        self.assertEqual(top_k([1, 2], 0), [])
        with self.assertRaises(ValueError):
            top_k([1], -1)

    def test_bfs_shortest_path(self) -> None:
        graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
        self.assertEqual(bfs_distance(graph, "a", "d"), 2)
        self.assertIsNone(bfs_distance(graph, "d", "a"))
        self.assertEqual(bfs_distance(graph, "a", "a"), 0)

    def test_lru_eviction_and_refresh(self) -> None:
        cache = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.get("a"), 1)
        cache.put("c", 3)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)
        cache.put("a", 4)
        self.assertEqual(cache.get("a"), 4)


if __name__ == "__main__":
    unittest.main()
