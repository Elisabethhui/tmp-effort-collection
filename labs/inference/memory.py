"""Dependency-free KV-cache memory calculations."""

from __future__ import annotations


def kv_cache_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """Return bytes for K and V cache under a simple dense-cache model."""

    values = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "bytes_per_element": bytes_per_element,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError("all cache dimensions and dtype size must be positive")
    # 2 accounts for the K and V tensors.
    return 2 * batch_size * sequence_length * num_layers * num_kv_heads * head_dim * bytes_per_element


def bytes_to_gib(num_bytes: int) -> float:
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative")
    return num_bytes / (1024**3)

