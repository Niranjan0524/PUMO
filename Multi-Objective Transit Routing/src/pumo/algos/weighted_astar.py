"""B0: exact weighted-sum search with a time-dependent label frontier."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time

from pumo.model import Journey, Query, RouteResult, validate_result
from pumo.objectives import OBJECTIVES, weighted_score
from .base import Router, edge_arrival, make_legs, query_start, step_values


@dataclass
class _ScalarLabel:
    node: int
    arrival: int
    values: dict[str, float]
    last_trip: str | None
    on_trip: str | None
    parent: int
    edge: object | None
    score: float
    alive: bool = True


class WeightedAStarRouter(Router):
    """Return the optimum for one non-negative weighted-sum profile.

    Since an earlier label may catch a better departure, each vehicle state
    retains the non-dominated (arrival, scalar-score) frontier.
    """

    name = "B0"

    def __init__(self, network=None, *, scales=None, label_cap=2_000_000,
                 max_speed_cap_mps=40.0):
        super().__init__(network, label_cap=label_cap)
        self.scales = dict(scales or {key: 1.0 for key in OBJECTIVES})
        self.max_speed_cap_mps = max_speed_cap_mps

    def route(self, q: Query) -> RouteResult:
        if self.network is None:
            raise ValueError("router has no PreparedNetwork")
        if q.profile_weights is None:
            raise ValueError("B0 requires profile_weights")
        started = time.perf_counter()
        deadline = started + float(q.timeout_s)
        net = self.network
        source, departure = query_start(q, net)
        target = net.nearest_node(q.destination)
        weights = {key: float(q.profile_weights.get(key, 0.0)) for key in q.objectives}
        if not all(math.isfinite(weight) and weight >= 0 for weight in weights.values()):
            raise ValueError("B0 weights must be finite and non-negative")
        if not any(weight > 0 for weight in weights.values()):
            raise ValueError("B0 requires at least one positive selected-objective weight")
        # Validate scale completeness and positivity before entering the search.
        weighted_score({key: 0.0 for key in q.objectives}, weights, self.scales, q.objectives)

        observed_speed = _max_speed(net)
        cap = self.max_speed_cap_mps
        if cap is not None and (not math.isfinite(float(cap)) or float(cap) <= 0):
            raise ValueError("max_speed_cap_mps must be positive and finite")
        # If the observed maximum exceeds the configured cap, disable the time
        # heuristic. Using the smaller cap would overestimate and break A*.
        heuristic_speed = observed_speed if cap is None or observed_speed <= float(cap) else None

        def scalar_score(values):
            elapsed = dict(values)
            elapsed["T"] = max(0.0, float(values.get("T", departure)) - departure)
            return weighted_score(elapsed, weights, self.scales, q.objectives)

        def heuristic(node):
            return time_lower_bound(net, node, target, weights, self.scales,
                                    heuristic_speed, q.objectives)

        initial = {name: 0.0 for name in OBJECTIVES}
        initial["T"] = float(departure)
        labels = [_ScalarLabel(source, departure, initial, None, None, -1, None, 0.0)]
        bags = {(source, None, None): [0]}
        queue = [(heuristic(source), 0.0, departure, 0)]
        best_goal = None
        generated = expanded = 0
        timed_out = cap_hit = False

        while queue:
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            f_score, _, _, index = heapq.heappop(queue)
            label = labels[index]
            if not label.alive:
                continue
            if best_goal is not None and f_score >= labels[best_goal].score - 1e-12:
                break
            expanded += 1
            if label.node == target:
                if best_goal is None or _better_goal(label, labels[best_goal]):
                    best_goal = index
                continue

            for edge in net.transitions_from(label.node, label.arrival, label.on_trip,
                                             disruptions=q.disruptions, deadline=deadline):
                arrival = edge_arrival(edge, label.arrival, on_trip=label.on_trip)
                if arrival is None or arrival <= label.arrival:
                    continue
                trip = edge.trip_id if edge.mode != "WALK" else None
                last_trip = edge.trip_id if trip else label.last_trip
                values = step_values(label.values, edge, label.arrival, arrival,
                                     previous_trip=label.last_trip, on_trip=label.on_trip,
                                     network=net)
                if values.get("X", 0.0) > q.max_transfers:
                    continue
                candidate_score = scalar_score(values)
                key = (edge.to_node, last_trip, trip)
                current = bags.get(key, ())
                if any(labels[j].alive and labels[j].arrival <= arrival and
                       labels[j].score <= candidate_score + 1e-12 for j in current):
                    continue
                for j in current:
                    if (labels[j].alive and arrival <= labels[j].arrival and
                            candidate_score <= labels[j].score + 1e-12):
                        labels[j].alive = False

                next_index = len(labels)
                labels.append(_ScalarLabel(edge.to_node, arrival, values, last_trip, trip,
                                           index, edge, candidate_score))
                generated += 1
                if edge.to_node == target:
                    if best_goal is None or _better_goal(labels[next_index], labels[best_goal]):
                        best_goal = next_index
                else:
                    bags.setdefault(key, []).append(next_index)
                    heapq.heappush(queue, (candidate_score + heuristic(edge.to_node),
                                           candidate_score, arrival, next_index))
                if generated >= self.label_cap:
                    cap_hit = True
                    break
            if time.perf_counter() >= deadline: timed_out = True
            if cap_hit or timed_out:
                break

        journeys = []
        if best_goal is not None:
            transitions = []
            cursor = best_goal
            while labels[cursor].parent >= 0:
                parent = labels[cursor].parent
                edge = labels[cursor].edge
                contribution = {key: labels[cursor].values.get(key, 0.0) -
                                labels[parent].values.get(key, 0.0)
                                for key in OBJECTIVES if key != "T"}
                contribution["T"] = labels[cursor].arrival - labels[parent].arrival
                transitions.append((edge, labels[cursor].arrival, contribution))
                cursor = parent
            transitions.reverse()
            totals = dict(labels[best_goal].values)
            totals["T"] = labels[best_goal].arrival - departure
            journeys = [Journey(make_legs(transitions, departure), totals)]

        result = RouteResult(journeys, {
            "runtime_s": time.perf_counter() - started,
            "labels_generated": generated,
            "labels_expanded": expanded,
            "timed_out": timed_out,
            "label_cap_hit": cap_hit,
            "exact": not timed_out and not cap_hit,
            "weighted_score": labels[best_goal].score if best_goal is not None else None,
            "observed_max_speed_mps": observed_speed,
            "heuristic_speed_bound_mps": heuristic_speed,
            "distance_bound_scale": getattr(net, "_distance_bound_scale", None),
            "heuristic_enabled": heuristic_speed is not None and "T" in q.objectives and weights.get("T", 0.0) > 0,
        })
        validate_result(result, objectives=q.objectives, pareto=False)
        return result


def _better_goal(candidate: _ScalarLabel, incumbent: _ScalarLabel) -> bool:
    if candidate.score < incumbent.score - 1e-12:
        return True
    return abs(candidate.score - incumbent.score) <= 1e-12 and candidate.arrival < incumbent.arrival


def _max_speed(net):
    """Maximum observed positive edge speed, in metres per second."""
    cached = getattr(net, "max_observed_speed_mps", None)
    if cached is not None:
        return float(cached)
    maximum = 0.0
    for adjacency in (net.walk, net.transit):
        for edges in adjacency.values():
            for edge in edges:
                if edge.mode == "WALK":
                    duration = edge.walk_time_s
                elif edge.departure_time is not None and edge.arrival_time is not None:
                    duration = edge.arrival_time - edge.departure_time
                else:
                    continue
                if duration > 0 and edge.distance_m > 0:
                    maximum = max(maximum, edge.distance_m / duration)
    # No defensible speed bound means a zero heuristic, represented by infinity.
    maximum = maximum if maximum > 0 and math.isfinite(maximum) else math.inf
    try:
        net._observed_max_speed_mps = maximum
    except (AttributeError, TypeError):
        pass
    return maximum


def time_lower_bound(net, node, target, weights, scales, max_speed, objectives=OBJECTIVES):
    """Admissible remaining scalar-score bound; all non-time terms are zero."""
    if "T" not in objectives or float(weights.get("T", 0.0)) == 0:
        return 0.0
    if max_speed is None or not math.isfinite(float(max_speed)) or float(max_speed) <= 0:
        return 0.0
    distance = net.straight_line_lower_bound_m(int(node), int(target))
    scale = float(scales["T"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale for T must be positive and finite")
    return float(weights["T"]) * distance / float(max_speed) / scale
