"""Objective vector conventions and independent journey aggregation."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite

OBJECTIVES = ("T", "C", "X", "W", "E", "S")


def objective_vector(values: Mapping[str, float], objectives: Iterable[str] = OBJECTIVES) -> tuple[float, ...]:
    names = tuple(objectives)
    unknown = set(names) - set(OBJECTIVES)
    if unknown:
        raise ValueError(f"unknown objectives: {sorted(unknown)}")
    if not names or len(set(names)) != len(names):
        raise ValueError("objectives must be a non-empty sequence without duplicates")
    vector = tuple(float(values.get(name, 0.0)) for name in names)
    if not all(isfinite(value) for value in vector):
        raise ValueError("objective values must be finite")
    return vector


def sum_leg_vectors(legs, objectives: Iterable[str] = OBJECTIVES) -> dict[str, float]:
    """Recompute objective totals from per-leg contributions, including wait time."""
    result = {name: 0.0 for name in objectives}
    # T is always retained because validators and callers need elapsed time,
    # even when the requested Pareto projection omits it.
    result.setdefault("T", 0.0)
    legs = list(legs)
    for leg in legs:
        for name in result:
            if name != "T":
                result[name] += float(leg.vector.get(name, 0.0))
        result["T"] += float(leg.vector.get("T", 0.0))
    return result


def normalize(values: Mapping[str, float], scales: Mapping[str, float], objectives=OBJECTIVES) -> dict[str, float]:
    output = {}
    for name in objectives:
        if name not in scales:
            raise ValueError(f"missing scale for objective {name}")
        scale = float(scales[name])
        if not isfinite(scale) or scale <= 0:
            raise ValueError(f"scale for {name} must be positive")
        value = float(values.get(name, 0.0))
        if not isfinite(value):
            raise ValueError(f"value for {name} must be finite")
        output[name] = value / scale
    return output


def weighted_score(values: Mapping[str, float], weights: Mapping[str, float],
                   scales: Mapping[str, float], objectives=OBJECTIVES) -> float:
    names = tuple(objectives)
    normalized = normalize(values, scales, names)
    total = 0.0
    for key in names:
        weight = float(weights.get(key, 0.0))
        if not isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {key} must be finite and non-negative")
        total += weight * normalized[key]
    return total


def profile_regret(candidates, reference, weights, scales, objectives=OBJECTIVES) -> float:
    if not candidates or not reference:
        return float("nan")
    score = lambda v: weighted_score(v, weights, scales, objectives)
    best_ref = min(map(score, reference))
    best = min(map(score, candidates))
    return (best - best_ref) / max(abs(best_ref), 1e-12)


def chebyshev_score(values, weights, scales, objectives=OBJECTIVES) -> float:
    return max((float(weights.get(k, 0.0)) * float(values.get(k, 0.0)) / float(scales[k])
                for k in objectives), default=0.0)
