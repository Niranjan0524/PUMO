"""Shared Pareto dominance and filtering for minimized objective vectors."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import isfinite
from typing import TypeVar

T = TypeVar("T")


def dominates(a: Sequence[float], b: Sequence[float], *, eps: float = 0.0) -> bool:
    """Return whether vector ``a`` weakly improves every component of ``b``.

    At least one component must improve by more than ``eps``. Vectors must have
    equal dimensions and contain finite values.
    """
    if len(a) != len(b):
        raise ValueError("objective vectors must have equal length")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    av, bv = tuple(map(float, a)), tuple(map(float, b))
    if not all(isfinite(x) for x in av + bv):
        raise ValueError("objective vectors must be finite")
    return all(x <= y + eps for x, y in zip(av, bv)) and any(x < y - eps for x, y in zip(av, bv))


def pareto_indices(vectors: Sequence[Sequence[float]], *, eps: float = 0.0) -> list[int]:
    """Indices of the first occurrence of each unique non-dominated vector."""
    keep: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for i, vector in enumerate(vectors):
        key = tuple(float(x) for x in vector)
        if key in seen:
            continue
        seen.add(key)
        if any(dominates(other, key, eps=eps) for j, other in enumerate(vectors) if j != i):
            continue
        keep.append(i)
    return keep


def pareto_filter(items: Iterable[T], vector_of, *, eps: float = 0.0) -> list[T]:
    """Filter objects by a vector extractor, retaining stable input order."""
    values = list(items)
    return [values[i] for i in pareto_indices([vector_of(v) for v in values], eps=eps)]


def epsilon_match(a: Sequence[float], b: Sequence[float], *, relative: float = 0.01,
                  absolute: float = 1e-9) -> bool:
    """Whether two vectors match within per-component relative tolerance."""
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= max(absolute, relative * max(abs(float(x)), abs(float(y))))
               for x, y in zip(a, b))
