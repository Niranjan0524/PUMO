"""Stable shared interfaces used by all routers and the benchmark runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


def validate_journey(journey: Journey, *, tolerance: float = 1e-6) -> None:
    """Basic shared leg continuity/time/vector checks."""
    for previous, current in zip(journey.legs, journey.legs[1:]):
        if previous.to_ref != current.from_ref:
            raise ValueError(f"disconnected legs: {previous.to_ref!r} != {current.from_ref!r}")
        if current.dep_time < previous.arr_time:
            raise ValueError("journey time moves backwards")
    recomputed = {}
    for leg in journey.legs:
        if leg.arr_time < leg.dep_time:
            raise ValueError("leg arrives before it departs")
        for key, value in leg.vector.items():
            if key != "T": recomputed[key] = recomputed.get(key, 0.0) + float(value)
        recomputed["T"] = recomputed.get("T", 0.0) + float(leg.vector.get("T", 0.0))
    for key, value in journey.vector.items():
        if abs(float(value) - recomputed.get(key, 0.0)) > tolerance:
            raise ValueError(f"reported {key}={value} disagrees with legs ({recomputed.get(key,0.0)})")
