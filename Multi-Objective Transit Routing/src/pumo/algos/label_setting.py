"""A1: exact Pareto label-setting on the configured time-dependent graph."""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import time

from pumo.model import Journey, Query, RouteResult, validate_result
from pumo.objectives import OBJECTIVES
from pumo.pareto import (pareto_filter, time_aware_dominates,
                          time_aware_equivalent)
from .base import Router, edge_arrival, make_legs, query_start, step_values


@dataclass
class _Label:
    node: int
    arrival: int
    values: dict[str, float]
    last_trip: str | None
    on_trip: str | None
    parent: int
    edge: object | None
    alive: bool = True


class LabelSettingRouter(Router):
    name = "A1"

    def route(self, q: Query) -> RouteResult:
        if self.network is None: raise ValueError("router has no PreparedNetwork")
        if getattr(self.network, "scenario", "core") not in {"core", "toy"}:
            raise ValueError("A1 is restricted to the Core scenario")
        started = time.perf_counter()
        deadline = started + float(q.timeout_s)
        net = self.network
        source, departure = query_start(q, net); target = net.nearest_node(q.destination)
        initial = {name: 0.0 for name in OBJECTIVES}; initial["T"] = float(departure)
        labels = [_Label(source, departure, initial, None, None, -1, None)]
        bags: dict[tuple, list[int]] = {(source, None, None): [0]}
        queue = [(departure, 0)]
        goals: list[int] = []
        generated = expanded = 0
        timed_out = cap_hit = False
        while queue:
            if time.perf_counter() >= deadline:
                timed_out = True; break
            _, index = heapq.heappop(queue)
            label = labels[index]
            if not label.alive: continue
            if any(_dominates_label(labels[g], label, q.objectives) for g in goals if labels[g].alive):
                continue
            expanded += 1
            if label.node == target:
                _add_goal(goals, labels, index, q.objectives)
                continue
            for edge in net.transitions_from(label.node, label.arrival, label.on_trip,
                                             disruptions=q.disruptions, deadline=deadline):
                arrival = edge_arrival(edge, label.arrival, on_trip=label.on_trip)
                if arrival is None or arrival <= label.arrival: continue
                trip = edge.trip_id if edge.mode != "WALK" else None
                last_trip = edge.trip_id if trip else label.last_trip
                values = step_values(label.values, edge, label.arrival, arrival,
                                     previous_trip=label.last_trip, on_trip=label.on_trip, network=net)
                if values.get("X",0.0) > q.max_transfers: continue
                if edge.mode != "WALK" and edge.trip_id != label.on_trip:
                    # The first trip has no transfer; X already reflects the previous trip.
                    pass
                key = (edge.to_node, last_trip, trip)
                proposed = _Label(edge.to_node, arrival, values, last_trip, trip, index, edge)
                if any(_dominates_label(labels[g], proposed, q.objectives) for g in goals if labels[g].alive):
                    continue
                if edge.to_node == target:
                    idx = len(labels)
                    labels.append(proposed)
                    generated += 1
                    _add_goal(goals, labels, idx, q.objectives)
                    if generated >= self.label_cap:
                        cap_hit = True; break
                    continue
                if any(_dominates_or_equal_label(labels[j], proposed, q.objectives)
                       for j in bags.get(key, ()) if labels[j].alive):
                    continue
                for j in bags.get(key, ()):
                    if labels[j].alive and _dominates_label(proposed, labels[j], q.objectives):
                        labels[j].alive = False
                idx = len(labels)
                labels.append(proposed)
                bags.setdefault(key, []).append(idx)
                generated += 1
                if generated >= self.label_cap:
                    cap_hit = True; break
                heapq.heappush(queue, (arrival, idx))
            if time.perf_counter() >= deadline: timed_out = True
            if cap_hit or timed_out: break
        # A cap or timeout can stop with an undispatched target label in the
        # queue. Include every target label already discovered in the partial
        # result, then apply the same shared Pareto filter as completed runs.
        goals = list(dict.fromkeys(goals + [i for i, label in enumerate(labels)
                                            if label.alive and label.node == target]))
        journeys = []
        for goal in goals:
            transitions=[]; cursor=goal
            while labels[cursor].parent >= 0:
                parent=labels[cursor].parent
                edge=labels[cursor].edge
                contribution={k: labels[cursor].values.get(k,0.0)-labels[parent].values.get(k,0.0)
                              for k in OBJECTIVES if k != "T"}
                contribution["T"] = labels[cursor].arrival-labels[parent].arrival
                transitions.append((edge, labels[cursor].arrival, contribution)); cursor=parent
            transitions.reverse()
            legs=make_legs(transitions, departure)
            totals=dict(labels[goal].values); totals["T"]=labels[goal].arrival-departure
            journeys.append(Journey(legs, totals))
        journeys = pareto_filter(journeys, lambda j: tuple(j.vector.get(k, 0.0) for k in q.objectives),
                                 eps=1e-12)
        result = RouteResult(journeys, {"runtime_s": time.perf_counter()-started,
            "labels_generated": generated, "labels_expanded": expanded, "timed_out": timed_out,
            "label_cap_hit": cap_hit, "exact": not timed_out and not cap_hit,
            "notes": ["partial Pareto set" if timed_out or cap_hit else "exact Pareto set"]})
        validate_result(result, objectives=q.objectives, pareto=True)
        return result


def _dominates_label(a, b, names):
    return time_aware_dominates(a.arrival, a.values, b.arrival, b.values, names, eps=1e-12)
def _dominates_or_equal_label(a, b, names):
    return time_aware_equivalent(a.arrival, a.values, b.arrival, b.values, names, eps=1e-12) or _dominates_label(a,b,names)


def _add_goal(goals, labels, index, objectives):
    candidate = labels[index]
    if any(_dominates_or_equal_label(labels[goal], candidate, objectives)
           for goal in goals if labels[goal].alive):
        candidate.alive = False
        return
    for goal in goals:
        if labels[goal].alive and _dominates_label(candidate, labels[goal], objectives):
            labels[goal].alive = False
    goals.append(index)
