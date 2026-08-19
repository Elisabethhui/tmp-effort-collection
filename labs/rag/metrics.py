"""Dependency-free retrieval metrics for interview experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence


def recall_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of relevant items found in the first k results."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


def reciprocal_rank(relevant: set[str], retrieved: Sequence[str], k: int | None = None) -> float:
    """Return the reciprocal rank of the first relevant result."""

    limit = len(retrieved) if k is None else max(k, 0)
    for index, item in enumerate(retrieved[:limit], start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def _dcg(gains: Sequence[float], k: int) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains[:k]))


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Compute NDCG@K for graded relevance scores in retrieved order."""

    if k <= 0:
        raise ValueError("k must be positive")
    actual = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0.0 else 0.0


def rrf_fuse(rankings: Iterable[Sequence[str]], *, rank_constant: int = 60) -> list[str]:
    """Fuse ranked document IDs with Reciprocal Rank Fusion.

    Ties are resolved by first appearance across the input rankings so the
    result is deterministic and easy to inspect in an interview experiment.
    """

    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            if document_id not in first_seen:
                first_seen[document_id] = seen_counter
                seen_counter += 1
            scores[document_id] += 1.0 / (rank_constant + rank)
    return sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
