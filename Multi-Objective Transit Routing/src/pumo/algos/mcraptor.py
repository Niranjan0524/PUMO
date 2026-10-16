"""A2: round-based, multi-criteria trip scanning with walking transfers.

This reference implementation scans the shared connection adjacency by trip
round. Street-graph walking is limited to origin access and destination egress;
between transit rides it uses the prepared Pareto footpath options.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time

from pumo.model import Journey, Query, RouteResult, validate_result
from pumo.objectives import OBJECTIVES
from pumo.pareto import (dominates, pareto_filter, time_aware_dominates,
                         time_aware_equivalent)
from .base import Router, edge_arrival, make_legs, query_start, step_values


@dataclass
class _RoundLabel:
    node: int
    arrival: int
    values: dict[str, float]
    boardings: int
    last_trip: str | None
    parent: int
    edge: object | None
    alive: bool = True


@dataclass
class _WalkLabel:
    node: int
    walk_time_s: int
    distance_m: float
    risk: float
    parent: int
    edge: object | None


@dataclass(frozen=True)
class _WalkOption:
    target: int
    walk_time_s: int
    distance_m: float
    risk: float
    edges: tuple[object, ...]


class McRaptorRouter(Router):
    """Round-based transit search with Pareto bags at each stop and round."""

    name = "A2"

    def __init__(self, network=None, *, label_cap=2_000_000,
                 access_radius_m=600.0, access_time_limit_s=900,
                 walking_option_limit=3, walking_label_cap=12):
        super().__init__(network, label_cap=label_cap)
        self.access_radius_m = float(access_radius_m)
        self.access_time_limit_s = int(access_time_limit_s)
        self.walking_option_limit = int(walking_option_limit)
        self.walking_label_cap = int(walking_label_cap)
        if self.access_radius_m <= 0 or self.access_time_limit_s <= 0:
            raise ValueError("access radius and walking time limit must be positive")
        if self.walking_option_limit <= 0 or self.walking_label_cap <= 0:
            raise ValueError("walking option and label limits must be positive")

    def route(self, q: Query) -> RouteResult:
        if self.network is None:
            raise ValueError("router has no PreparedNetwork")
        started = time.perf_counter()
        net = self.network
        source, departure = query_start(q, net)
        target = net.nearest_node(q.destination)
        deadline = started + float(q.timeout_s)
        budget = {"timed_out": False, "walking_label_cap_hit": False,
                  "walking_search_labels": 0, "walking_labels_expanded": 0}

        initial = {key: 0.0 for key in OBJECTIVES}
        initial["T"] = float(departure)
        labels = [_RoundLabel(source, departure, initial, 0, None, -1, None)]
        round_bags: list[dict[int, list[int]]] = [{source: [0]}]
        goals: list[int] = []
        generated = expanded = 0
        cap_hit = False
        egress_cache: dict[int, list[_WalkOption]] = {}
        egress_nodes = _nearby_stop_nodes(net, target, self.access_radius_m)
        access_nodes = _nearby_stop_nodes(net, source, self.access_radius_m)
        access_nodes.add(source)
        access_options = {}

        def add_edge(parent_index, edge, arrival, boardings, *, previous_trip=None,
                     on_trip=None):
            nonlocal generated, cap_hit
            parent = labels[parent_index]
            values = step_values(parent.values, edge, parent.arrival, arrival,
                                 previous_trip=parent.last_trip if previous_trip is None else previous_trip,
                                 on_trip=on_trip, network=net)
            last_trip = edge.trip_id if edge.mode != "WALK" else parent.last_trip
            index = len(labels)
            labels.append(_RoundLabel(int(edge.to_node), int(arrival), values,
                                      int(boardings), last_trip, parent_index, edge))
            generated += 1
            if generated >= self.label_cap:
                cap_hit = True
            return index

        def insert_round(round_index, index):
            label = labels[index]
            bag = round_bags[round_index].setdefault(label.node, [])
            if any(labels[old].alive and _label_dominates_or_equal(
                    labels[old], label, q.objectives) for old in bag):
                label.alive = False
                return False
            for old in bag:
                if labels[old].alive and _label_dominates(label, labels[old], q.objectives):
                    labels[old].alive = False
            bag.append(index)
            return True

        def is_pruned(index):
            label = labels[index]
            return any(_label_dominates_or_equal(labels[goal], label, q.objectives)
                       for goal in goals if labels[goal].alive)

        def record_goal(index):
            candidate = labels[index]
            if any(_label_dominates_or_equal(labels[goal], candidate, q.objectives)
                   for goal in goals if labels[goal].alive):
                candidate.alive = False
                return
            for goal in goals:
                if labels[goal].alive and _label_dominates(candidate, labels[goal], q.objectives):
                    labels[goal].alive = False
            goals.append(index)

        def walking_options(start_node, target_nodes, max_distance, max_time):
            return _bounded_walk_options(
                net, start_node, target_nodes, max_distance_m=max_distance,
                max_time_s=max_time, limit=self.walking_option_limit,
                labels_per_node=self.walking_label_cap, deadline=deadline, budget=budget)

        def extend_walk_path(parent_index, option, boardings):
            index = parent_index
            for edge in option.edges:
                arrival = edge_arrival(edge, labels[index].arrival, on_trip=None)
                if arrival is None or arrival <= labels[index].arrival:
                    continue
                index = add_edge(index, edge, arrival, boardings)
                if cap_hit:
                    break
            return index

        def egress_options(node):
            if node not in egress_cache:
                if node == target:
                    egress_cache[node] = [_WalkOption(target, 0, 0.0, 0.0, ())]
                elif node not in egress_nodes:
                    egress_cache[node] = []
                else:
                    options, _ = walking_options(node, {target}, self.access_radius_m,
                                                 self.access_time_limit_s)
                    egress_cache[node] = options
            return egress_cache[node]

        def add_egress_goals(index):
            if is_pruned(index):
                return
            for option in egress_options(labels[index].node):
                goal = extend_walk_path(index, option, labels[index].boardings)
                if labels[goal].node == target:
                    record_goal(goal)
                if cap_hit or budget["timed_out"]:
                    break

        # Round zero: origin access paths to nearby stops.
        access_options, _ = walking_options(source, access_nodes, self.access_radius_m,
                                           self.access_time_limit_s)
        for option in access_options:
            if option.target == source:
                continue
            index = extend_walk_path(0, option, 0)
            if labels[index].node in access_nodes:
                insert_round(0, index)
                add_egress_goals(index)
            if cap_hit or budget["timed_out"]:
                break
        add_egress_goals(0)

        max_boardings = q.max_transfers + 1
        highest_round = 0
        for round_index in range(1, max_boardings + 1):
            if cap_hit or budget["timed_out"] or time.perf_counter() >= deadline:
                budget["timed_out"] = budget["timed_out"] or time.perf_counter() >= deadline
                break
            round_bags.append({})
            previous_bag = round_bags[round_index - 1]
            for node, prior_indices in list(previous_bag.items()):
                if time.perf_counter() >= deadline:
                    budget["timed_out"] = True
                    break
                for prior_index in list(prior_indices):
                    if time.perf_counter() >= deadline:
                        budget["timed_out"] = True
                        break
                    if not labels[prior_index].alive or is_pruned(prior_index):
                        continue
                    expanded += 1
                    prior = labels[prior_index]
                    board_edges = [edge for edge in net.transitions_from(
                        node, prior.arrival, None, disruptions=q.disruptions, deadline=deadline)
                        if edge.mode != "WALK"]
                    for first_edge in board_edges:
                        if time.perf_counter() >= deadline:
                            budget["timed_out"] = True
                            break
                        arrival = edge_arrival(first_edge, prior.arrival, on_trip=None)
                        if arrival is None or arrival <= prior.arrival:
                            continue
                        trip_id = first_edge.trip_id
                        if trip_id is None:
                            continue
                        current_index = add_edge(prior_index, first_edge, arrival, round_index,
                                                 previous_trip=prior.last_trip, on_trip=None)
                        highest_round = max(highest_round, round_index)
                        insert_round(round_index, current_index)
                        add_egress_goals(current_index)

                        # Ride every consecutive connection for this trip.
                        visited = {(labels[current_index].node, labels[current_index].arrival)}
                        while not cap_hit and not budget["timed_out"]:
                            current = labels[current_index]
                            if is_pruned(current_index):
                                break
                            next_edges = [edge for edge in net.transitions_from(
                                current.node, current.arrival, trip_id,
                                disruptions=q.disruptions, deadline=deadline)
                                if edge.mode != "WALK" and edge.trip_id == trip_id]
                            if not next_edges:
                                break
                            next_edge = min(next_edges, key=lambda edge: (
                                int(edge.departure_time), int(edge.arrival_time), edge.segment_id))
                            next_arrival = edge_arrival(next_edge, current.arrival, on_trip=trip_id)
                            if next_arrival is None or next_arrival <= current.arrival:
                                break
                            visit = (int(next_edge.to_node), int(next_arrival))
                            if visit in visited:
                                break
                            visited.add(visit)
                            next_index = add_edge(current_index, next_edge, next_arrival,
                                                  round_index, previous_trip=trip_id,
                                                  on_trip=trip_id)
                            current_index = next_index
                            insert_round(round_index, current_index)
                            add_egress_goals(current_index)
                            if time.perf_counter() >= deadline:
                                budget["timed_out"] = True
                                break
                        if cap_hit or budget["timed_out"]:
                            break
                    if cap_hit or budget["timed_out"]:
                        break
                if cap_hit or budget["timed_out"]:
                    break

            # Relax cached stop-to-stop Pareto footpaths within this round.
            footpath_queue = []
            queued = set()
            for bag in round_bags[round_index].values():
                for index in bag:
                    if labels[index].alive:
                        heapq.heappush(footpath_queue, (labels[index].arrival, index))
                        queued.add(index)
            while footpath_queue and not cap_hit and not budget["timed_out"]:
                _, index = heapq.heappop(footpath_queue)
                label = labels[index]
                if not label.alive or is_pruned(index):
                    continue
                expanded += 1
                for edge in _footpath_edges(net, label.node, self.access_radius_m):
                    arrival = edge_arrival(edge, label.arrival, on_trip=None)
                    if arrival is None or arrival <= label.arrival:
                        continue
                    next_index = add_edge(index, edge, arrival, round_index)
                    if insert_round(round_index, next_index):
                        add_egress_goals(next_index)
                        if next_index not in queued:
                            heapq.heappush(footpath_queue, (arrival, next_index))
                            queued.add(next_index)
                    if cap_hit or budget["timed_out"]:
                        break
                if time.perf_counter() >= deadline:
                    budget["timed_out"] = True
            if not round_bags[round_index]:
                break

        # Include a bounded walking-only alternative. It is deliberately run
        # after transit rounds so a long pedestrian search cannot starve them.
        if not cap_hit and not budget["timed_out"] and time.perf_counter() < deadline:
            straight_distance = net.straight_line_distance_m(source, target)
            walk_limit = max(1000.0, straight_distance * 3.0)
            walk_time_limit = max(3600, int(walk_limit / 0.5))
            direct_options, _ = walking_options(source, {target}, walk_limit, walk_time_limit)
            for option in direct_options:
                index = extend_walk_path(0, option, 0)
                if labels[index].node == target:
                    record_goal(index)
                if cap_hit or budget["timed_out"]:
                    break

        journeys = [_journey_from_label(labels, goal, departure) for goal in goals if labels[goal].alive]
        journeys = pareto_filter(journeys,
                                 lambda journey: tuple(journey.vector.get(key, 0.0)
                                                       for key in q.objectives), eps=1e-6)
        runtime = time.perf_counter() - started
        timed_out = budget["timed_out"] or runtime >= q.timeout_s
        result = RouteResult(journeys, {
            "runtime_s": runtime,
            "labels_generated": generated,
            "labels_expanded": expanded,
            "rounds": highest_round,
            "timed_out": timed_out,
            "label_cap_hit": cap_hit or budget["walking_label_cap_hit"],
            "walking_label_cap_hit": budget["walking_label_cap_hit"],
            "walking_search_labels": budget["walking_search_labels"],
            "walking_labels_expanded": budget["walking_labels_expanded"],
            "access_stops": len(access_nodes),
            "egress_stops": len(egress_nodes),
            "footpath_transitions": sum(len(v) for v in net.footpaths.values()),
            "exact": not timed_out and not cap_hit and not budget["walking_label_cap_hit"],
            "implementation_note": "round-based connection trip scan; street graph is used for bounded access/egress only",
            "walking_model_note": "access and egress keep at most configured K options; transfers use prepared footpaths",
        })
        validate_result(result, objectives=q.objectives, pareto=True, tolerance=1e-6)
        return result


def _label_dominates(a, b, objectives):
    return time_aware_dominates(a.arrival, a.values, b.arrival, b.values,
                                objectives, eps=1e-12)


def _label_dominates_or_equal(a, b, objectives):
    return (time_aware_equivalent(a.arrival, a.values, b.arrival, b.values,
                                  objectives, eps=1e-12) or
            _label_dominates(a, b, objectives))


def _journey_from_label(labels, goal, departure):
    transitions = []
    cursor = goal
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
    totals = dict(labels[goal].values)
    totals["T"] = labels[goal].arrival - departure
    return Journey(make_legs(transitions, departure), totals)


def _nearby_stop_nodes(net, center, radius_m):
    candidates = set(net.node_stops)
    if not candidates:
        candidates = {node for node, edges in net.transit.items() if edges}
        candidates.update(edge.to_node for edges in net.transit.values() for edge in edges)
    return {int(node) for node in candidates
            if node in net.coords and net.straight_line_distance_m(center, node) <= radius_m}


def _footpath_edges(net, node, fallback_radius_m):
    if net.footpaths:
        return net.footpaths.get(int(node), ())
    # Toy-network fallback: permit one explicitly stored street edge between
    # stop nodes when no precomputed footpath product exists.
    stop_nodes = set(net.node_stops)
    if not stop_nodes:
        stop_nodes = {source for source, edges in net.transit.items() if edges}
        stop_nodes.update(edge.to_node for edges in net.transit.values() for edge in edges)
    return [edge for edge in net.walk.get(int(node), ())
            if edge.to_node in stop_nodes and edge.distance_m <= fallback_radius_m]


def _bounded_walk_options(net, source, targets, *, max_distance_m, max_time_s,
                          limit, labels_per_node, deadline, budget):
    targets = {int(node) for node in targets}
    source = int(source)
    if source not in net.coords or not targets:
        return [], False
    max_distance_m = max(0.0, float(max_distance_m))
    max_time_s = max(0, int(max_time_s))
    labels = [_WalkLabel(source, 0, 0.0, 0.0, -1, None)]
    active = {source: [0]}
    queue = [(0, 0)]
    truncated = False
    while queue:
        if time.perf_counter() >= deadline:
            budget["timed_out"] = True
            break
        _, index = heapq.heappop(queue)
        label = labels[index]
        if index not in active.get(label.node, ()):
            continue
        budget["walking_labels_expanded"] += 1
        for edge in net.walk.get(label.node, ()):
            walk_time = label.walk_time_s + int(edge.walk_time_s)
            distance = label.distance_m + float(edge.distance_m)
            risk = label.risk + float(edge.risk)
            if walk_time > max_time_s or distance > max_distance_m:
                continue
            candidate = (walk_time, distance, risk)
            current = active.setdefault(int(edge.to_node), [])
            if any(dominates((labels[old].walk_time_s, labels[old].distance_m, labels[old].risk),
                             candidate, eps=1e-9) or
                   _walk_equivalent(labels[old], candidate) for old in current):
                continue
            survivors = [old for old in current if not dominates(
                candidate, (labels[old].walk_time_s, labels[old].distance_m, labels[old].risk), eps=1e-9)]
            new_index = len(labels)
            labels.append(_WalkLabel(int(edge.to_node), walk_time, distance, risk, index, edge))
            budget["walking_search_labels"] += 1
            survivors.append(new_index)
            if len(survivors) > labels_per_node:
                truncated = True
                budget["walking_label_cap_hit"] = True
                risk_scale = max(1.0, max(labels[i].risk for i in survivors))
                survivors.sort(key=lambda i: (labels[i].walk_time_s / max(1, max_time_s) +
                                              labels[i].distance_m / max(1.0, max_distance_m) +
                                              labels[i].risk / risk_scale,
                                              labels[i].walk_time_s, labels[i].distance_m,
                                              labels[i].risk, i))
                survivors = survivors[:labels_per_node]
            active[int(edge.to_node)] = survivors
            if new_index in survivors:
                heapq.heappush(queue, (walk_time, new_index))

    results = []
    for target in sorted(targets):
        for index in sorted(active.get(target, ()), key=lambda i: (
                labels[i].walk_time_s, labels[i].distance_m, labels[i].risk))[:limit]:
            edges = []
            cursor = index
            while labels[cursor].parent >= 0:
                edges.append(labels[cursor].edge)
                cursor = labels[cursor].parent
            edges.reverse()
            label = labels[index]
            results.append(_WalkOption(target, label.walk_time_s, label.distance_m,
                                       label.risk, tuple(edges)))
    return results, truncated


def _walk_equivalent(label, candidate, eps=1e-9):
    return (abs(label.walk_time_s - candidate[0]) <= eps and
            abs(label.distance_m - candidate[1]) <= eps and
            abs(label.risk - candidate[2]) <= eps)
