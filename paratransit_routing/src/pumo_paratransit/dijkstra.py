"""Algorithm A1: Dijkstra shortest path router with memoization/caching."""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple
from .network import RoadGraph


class DijkstraRouter:
    """Computes shortest paths, travel times, and distances on a RoadGraph with caching."""

    def __init__(self, graph: RoadGraph, default_weight: str = "time"):
        self.graph = graph
        self.default_weight = default_weight  # "time" or "distance"
        # Cache key: (origin, destination, weight_criterion) -> (travel_time_sec, distance_km, path)
        self._cache: Dict[Tuple[str, str, str], Tuple[float, float, List[str]]] = {}

    def get_travel_time(self, origin: str, destination: str) -> float:
        """Returns the minimum travel time in seconds between origin and destination."""
        if origin == destination:
            return 0.0
        time_sec, _, _ = self.find_shortest_path(origin, destination, weight_by="time")
        return time_sec

    def get_distance(self, origin: str, destination: str) -> float:
        """Returns the network distance in km along the travel-time-optimal path."""
        if origin == destination:
            return 0.0
        _, dist_km, _ = self.find_shortest_path(origin, destination, weight_by="time")
        return dist_km

    def find_shortest_path(
        self,
        origin: str,
        destination: str,
        weight_by: str = "time"
    ) -> Tuple[float, float, List[str]]:
        """Finds the shortest path from origin to destination using Dijkstra's algorithm.

        Returns:
            Tuple of (travel_time_sec, distance_km, path_nodes)
        """
        if origin == destination:
            return 0.0, 0.0, [origin]

        cache_key = (origin, destination, weight_by)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if origin not in self.graph.nodes:
            raise KeyError(f"Origin node '{origin}' not found in road network")
        if destination not in self.graph.nodes:
            raise KeyError(f"Destination node '{destination}' not found in road network")

        # Priority queue stores tuples: (accumulated_cost, current_node, time_sec, dist_km, path)
        pq: List[Tuple[float, str, float, float, List[str]]] = [(0.0, origin, 0.0, 0.0, [origin])]
        best_cost: Dict[str, float] = {origin: 0.0}

        while pq:
            cost, u, t_sec, d_km, path = heapq.heappop(pq)

            if u == destination:
                result = (t_sec, d_km, path)
                self._cache[cache_key] = result
                return result

            if cost > best_cost.get(u, float("inf")):
                continue

            for edge in self.graph.get_neighbors(u):
                v = edge.to_node
                edge_cost = edge.travel_time_sec if weight_by == "time" else edge.distance_km
                new_cost = cost + edge_cost

                if new_cost < best_cost.get(v, float("inf")):
                    best_cost[v] = new_cost
                    new_t = t_sec + edge.travel_time_sec
                    new_d = d_km + edge.distance_km
                    heapq.heappush(pq, (new_cost, v, new_t, new_d, path + [v]))

        # No path found
        unreachable = (float("inf"), float("inf"), [])
        self._cache[cache_key] = unreachable
        return unreachable

    def precompute_all_pairs(self, weight_by: str = "time") -> None:
        """Precomputes all-pairs shortest paths for fast lookups in large simulation runs."""
        nodes = list(self.graph.nodes.keys())
        for u in nodes:
            # Run one single-source Dijkstra from u to all v
            dist_map: Dict[str, Tuple[float, float, List[str]]] = {u: (0.0, 0.0, [u])}
            pq: List[Tuple[float, str, float, float, List[str]]] = [(0.0, u, 0.0, 0.0, [u])]
            best_cost: Dict[str, float] = {u: 0.0}

            while pq:
                cost, curr, t_sec, d_km, path = heapq.heappop(pq)
                if cost > best_cost.get(curr, float("inf")):
                    continue

                for edge in self.graph.get_neighbors(curr):
                    v = edge.to_node
                    edge_cost = edge.travel_time_sec if weight_by == "time" else edge.distance_km
                    new_cost = cost + edge_cost

                    if new_cost < best_cost.get(v, float("inf")):
                        best_cost[v] = new_cost
                        new_t = t_sec + edge.travel_time_sec
                        new_d = d_km + edge.distance_km
                        new_path = path + [v]
                        dist_map[v] = (new_t, new_d, new_path)
                        heapq.heappush(pq, (new_cost, v, new_t, new_d, new_path))

            for v in nodes:
                res = dist_map.get(v, (float("inf"), float("inf"), []))
                self._cache[(u, v, weight_by)] = res
