"""Shared naming helpers for variable-length event sequences.

An anytime sample contains one context voxel immediately before the RGB frame
and one voxel for every forward warp.  The first three keys keep the legacy
names so existing two-warp checkpoints and data remain compatible.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping


def event_key_from_offset(offset: int) -> str:
    """Return the sample key for an event window relative to input RGB t0."""
    offset = int(offset)
    if offset == -1:
        return "event_before"
    if offset == 0:
        return "event"
    if offset == 1:
        return "event_after"
    if offset >= 2:
        return f"event_after_{offset}"
    raise ValueError(f"Event offset must be >= -1, got {offset}")


def event_offset_from_key(key: str) -> int:
    """Invert :func:`event_key_from_offset`."""
    if key == "event_before":
        return -1
    if key == "event":
        return 0
    if key == "event_after":
        return 1
    prefix = "event_after_"
    if key.startswith(prefix):
        offset = int(key[len(prefix):])
        if offset >= 2:
            return offset
    raise ValueError(f"Unsupported temporal event key: {key!r}")


def event_keys_for_warps(num_warps: int) -> List[str]:
    """Return context + forward event keys for ``num_warps`` flow warps."""
    num_warps = int(num_warps)
    if num_warps <= 0:
        raise ValueError(f"num_warps must be positive, got {num_warps}")
    return [event_key_from_offset(offset) for offset in range(-1, num_warps)]


def event_keys_in_mapping(mapping: Mapping[str, object]) -> List[str]:
    """Return all recognized event keys in chronological order."""
    pairs = []
    for key in mapping:
        try:
            pairs.append((event_offset_from_key(key), key))
        except (TypeError, ValueError):
            continue
    return [key for _, key in sorted(pairs)]


def validate_event_keys(keys: Iterable[str], num_warps: int) -> List[str]:
    """Validate that ``keys`` are the complete sequence expected by a model."""
    actual = list(keys)
    expected = event_keys_for_warps(num_warps)
    if actual != expected:
        raise ValueError(f"Expected temporal event keys {expected}, got {actual}")
    return actual
