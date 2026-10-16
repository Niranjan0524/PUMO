from __future__ import annotations
from abc import ABC, abstractmethod
import time

from pumo.model import Query, RouteResult
from pumo.network import PreparedNetwork, Transition


class Router(ABC):
    name = "router"

    def __init__(self, network: PreparedNetwork | None = None, *, label_cap: int = 2_000_000):
        self.network = network
        self.label_cap = int(label_cap)

    def prepare(self, net: PreparedNetwork, cfg: dict) -> dict:
        started = time.perf_counter()
        self.network = net
        return {"time_s": time.perf_counter() - started, "storage_mb": 0.0}

    @abstractmethod
    def route(self, q: Query) -> RouteResult:
        raise NotImplementedError


def query_start(q: Query, network: PreparedNetwork):
    """Resolve an ordinary origin or a mid-journey node/stop state."""
    if q.start_state is None:
        return network.nearest_node(q.origin), int(q.departure_time)
    ref=str(q.start_state.current_position)
    if ref.startswith("node:"): ref=ref.split(":",1)[1]
    elif ref.startswith("stop:"):
        stop=int(ref.split(":",1)[1])
        if stop not in network.stop_nodes: raise ValueError(f"unknown reroute stop {stop}")
        return network.stop_nodes[stop], int(q.start_state.current_time)
    try: node=int(ref)
    except ValueError as exc: raise ValueError("start_state.current_position must be a node ID or stop:<id>") from exc
    if node not in network.coords: raise ValueError(f"unknown reroute node {node}")
    return node, int(q.start_state.current_time)


def step_values(values, edge: Transition, current_time: int, arrival: int, *,
                previous_trip: str | None, on_trip: str | None,
                network: PreparedNetwork) -> dict[str, float]:
    out = dict(values)
    out["T"] = float(arrival)
    new_boarding = edge.mode != "WALK" and edge.trip_id != on_trip
    if edge.mode == "WALK":
        out["W"] = out.get("W", 0.0) + edge.distance_m
        out["S"] = out.get("S", 0.0) + edge.risk
    else:
        if new_boarding:
            out["C"] = out.get("C", 0.0) + float(network.fare_per_boarding.get(edge.mode, 0.0))
            if previous_trip is not None: out["X"] = out.get("X", 0.0) + 1.0
        out["C"] = out.get("C", 0.0) + edge.cost
        wait = max(0, int(edge.departure_time) - int(current_time))
        out["S"] = out.get("S", 0.0) + wait * float(getattr(network, "node_risk", {}).get(edge.from_node, 0.0))
    out["E"] = out.get("E", 0.0) + edge.emissions
    return out


def edge_arrival(edge: Transition, current_time: int, *, on_trip: str | None) -> int | None:
    if edge.mode == "WALK": return int(current_time) + int(edge.walk_time_s)
    if edge.trip_id == on_trip:
        return int(edge.arrival_time) if int(edge.departure_time) >= current_time else None
    return int(edge.arrival_time) if int(edge.departure_time) >= current_time else None


def make_legs(path, departure_time: int):
    """Reconstruct edge-level legs. Times include station waiting in transit legs."""
    from pumo.model import Leg
    legs = []
    current_time = int(departure_time)
    for edge, arrival, contribution in path:
        mode = edge.mode
        dep = current_time if mode == "WALK" else int(edge.departure_time)
        segment_ids = tuple(getattr(edge, "segment_ids", ())) or (edge.segment_id,)
        legs.append(Leg(mode, str(edge.from_node), str(edge.to_node), dep, int(arrival),
                        list(segment_ids), contribution))
        current_time = int(arrival)
    return legs
