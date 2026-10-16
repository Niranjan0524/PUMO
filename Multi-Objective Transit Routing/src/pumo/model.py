"""Stable shared interfaces used by all routers and the benchmark runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from .objectives import OBJECTIVES, objective_vector, sum_leg_vectors
from .pareto import dominates


@dataclass(frozen=True)
class Query:
    origin: tuple[float, float]
    destination: tuple[float, float]
    departure_time: int
    objectives: tuple[str, ...] = ("T", "C", "X", "W", "E", "S")
    profile_weights: dict[str, float] | None = None
    disruptions: tuple["Disruption", ...] = ()
    start_state: "JourneyState | None" = None
    max_transfers: int = 4
    timeout_s: float = 60.0

    def __post_init__(self):
        unknown = set(self.objectives) - set(OBJECTIVES)
        if not self.objectives or unknown:
            raise ValueError(f"objectives must be a non-empty subset of {OBJECTIVES}; unknown={sorted(unknown)}")
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("objectives must not contain duplicates")
        if self.max_transfers < 0:
            raise ValueError("max_transfers must be non-negative")
        if not isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")


@dataclass
class Leg:
    mode: str
    from_ref: str
    to_ref: str
    dep_time: int
    arr_time: int
    segment_ids: list[str] = field(default_factory=list)
    vector: dict[str, float] = field(default_factory=dict)


@dataclass
class Journey:
    legs: list[Leg]
    vector: dict[str, float]


@dataclass
class RouteResult:
    journeys: list[Journey]
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Disruption:
    event_id: str
    event_type: str
    start_time: int
    end_time: int
    edge_ids: tuple[int, ...] = ()
    trip_ids: tuple[str, ...] = ()
    time_penalty_s: int = 0
    closure: bool = False
    safety_penalty: float = 0.0
    source_type: str = "SYNTHETIC"
    seed: int = 0
    generator_version: str = "v1"


@dataclass(frozen=True)
class JourneyState:
    current_position: str
    current_time: int
    current_mode: str | None = None
    completed_legs: tuple[Leg, ...] = ()


def validate_journey(journey: Journey, *, tolerance: float = 1e-6,
                     objectives: tuple[str, ...] | None = None) -> None:
    """Validate leg continuity and independently recompute journey totals.

    T is accumulated from each leg's contribution so that a router can assign
    inter-leg waiting time to the following leg. The actual leg interval must
    still fit inside that contribution.
    """
    if tolerance < 0 or not isfinite(float(tolerance)):
        raise ValueError("tolerance must be non-negative and finite")
    objectives = tuple(journey.vector) if objectives is None else tuple(objectives)
    unknown = set(objectives) - set(OBJECTIVES)
    if not objectives or unknown or len(set(objectives)) != len(objectives):
        raise ValueError(f"invalid objective selection: {objectives}")
    for previous, current in zip(journey.legs, journey.legs[1:]):
        if previous.to_ref != current.from_ref:
            raise ValueError(f"disconnected legs: {previous.to_ref!r} != {current.from_ref!r}")
        if current.dep_time < previous.arr_time:
            raise ValueError("journey time moves backwards")
    recomputed = sum_leg_vectors(journey.legs, objectives)
    for leg in journey.legs:
        if leg.arr_time < leg.dep_time:
            raise ValueError("leg arrives before it departs")
        if leg.mode not in {"WALK", "BUS", "METRO"}:
            raise ValueError(f"unsupported leg mode: {leg.mode!r}")
        if float(leg.vector.get("T", 0.0)) + tolerance < leg.arr_time - leg.dep_time:
            raise ValueError("leg T contribution is shorter than its elapsed leg time")
        for key, value in leg.vector.items():
            if key not in OBJECTIVES:
                raise ValueError(f"unknown leg objective: {key}")
            if not isfinite(float(value)) or float(value) < -tolerance:
                raise ValueError(f"leg objective {key} must be finite and non-negative")
    for key in objectives:
        if key not in journey.vector:
            raise ValueError(f"journey is missing objective {key}")
        value = float(journey.vector[key])
        if not isfinite(value) or value < -tolerance:
            raise ValueError(f"journey objective {key} must be finite and non-negative")
        if abs(value - recomputed[key]) > tolerance:
            raise ValueError(f"reported {key}={value} disagrees with legs ({recomputed[key]})")


def validate_result(result: RouteResult, *, objectives: tuple[str, ...] = OBJECTIVES,
                    pareto: bool = True, tolerance: float = 1e-6) -> None:
    """Check every returned journey and, when requested, its Pareto property."""
    vectors = []
    for journey in result.journeys:
        validate_journey(journey, tolerance=tolerance, objectives=objectives)
        vectors.append(objective_vector(journey.vector, objectives))
    if pareto:
        for i, vector in enumerate(vectors):
            if any(dominates(other, vector, eps=tolerance) for j, other in enumerate(vectors) if j != i):
                raise ValueError("route result contains a dominated journey")
