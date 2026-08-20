"""Interview-sized data-structure implementations with explicit invariants."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Hashable, Iterable, Mapping, TypeVar


T = TypeVar("T")
NodeId = TypeVar("NodeId", bound=Hashable)


def top_k(values: Iterable[T], k: int) -> list[T]:
    """Return the k largest comparable values in descending order."""

    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return []

    heap: list[T] = []
    for value in values:
        if len(heap) < k:
            heapq.heappush(heap, value)
        elif value > heap[0]:
            heapq.heapreplace(heap, value)
    return sorted(heap, reverse=True)


def bfs_distance(graph: Mapping[NodeId, Iterable[NodeId]], start: NodeId, target: NodeId) -> int | None:
    """Return the unweighted shortest-path distance, or ``None`` if absent."""

    if start == target:
        return 0
    queue: deque[tuple[NodeId, int]] = deque([(start, 0)])
    visited = {start}
    while queue:
        node, distance = queue.popleft()
        for neighbor in graph.get(node, ()):
            if neighbor in visited:
                continue
            if neighbor == target:
                return distance + 1
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


@dataclass
class _Node:
    key: str
    value: object | None = None
    prev: "_Node | None" = None
    next: "_Node | None" = None


class LRUCache:
    """A small O(1) LRU cache using a hash map plus a doubly linked list."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: dict[str, _Node] = {}
        self._head = _Node("<head>")
        self._tail = _Node("<tail>")
        self._head.next = self._tail
        self._tail.prev = self._head

    def _detach(self, node: _Node) -> None:
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev

    def _attach_after_head(self, node: _Node) -> None:
        assert self._head.next is not None
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def get(self, key: str) -> object | None:
        node = self._items.get(key)
        if node is None:
            return None
        self._detach(node)
        self._attach_after_head(node)
        return node.value

    def put(self, key: str, value: object) -> None:
        node = self._items.get(key)
        if node is not None:
            node.value = value
            self._detach(node)
            self._attach_after_head(node)
            return

        node = _Node(key, value)
        self._items[key] = node
        self._attach_after_head(node)
        if len(self._items) > self.capacity:
            assert self._tail.prev is not None and self._tail.prev is not self._head
            evicted = self._tail.prev
            self._detach(evicted)
            del self._items[evicted.key]
