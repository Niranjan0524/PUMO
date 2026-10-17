"""Directed road network representation and network generators for PUMO paratransit."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import math


@dataclass
class Node:
    """Road intersection or activity hub."""
    node_id: str
    name: str
    x: float = 0.0  # Projected x coordinate (km or meters)
    y: float = 0.0  # Projected y coordinate (km or meters)
    zone: str = "urban"


@dataclass
class Edge:
    """Directed road segment."""
    from_node: str
    to_node: str
    distance_km: float
    free_flow_speed_kmh: float = 30.0
    travel_time_sec: float = 0.0
    road_type: str = "arterial"

    def __post_init__(self):
        if self.travel_time_sec <= 0.0:
            # travel time in seconds = (distance_km / speed_kmh) * 3600
            self.travel_time_sec = (self.distance_km / max(1.0, self.free_flow_speed_kmh)) * 3600.0


class RoadGraph:
    """Directed weighted road network for travel-time and distance routing."""

    def __init__(self, name: str = "RoadNetwork"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.adjacency: Dict[str, List[Edge]] = {}

    def add_node(self, node_id: str, name: str, x: float = 0.0, y: float = 0.0, zone: str = "urban") -> Node:
        node = Node(node_id=node_id, name=name, x=x, y=y, zone=zone)
        self.nodes[node_id] = node
        if node_id not in self.adjacency:
            self.adjacency[node_id] = []
        return node

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        distance_km: float,
        free_flow_speed_kmh: float = 30.0,
        travel_time_sec: Optional[float] = None,
        road_type: str = "arterial",
        bidirectional: bool = False,
    ) -> Edge:
        if from_node not in self.nodes:
            self.add_node(from_node, from_node)
        if to_node not in self.nodes:
            self.add_node(to_node, to_node)

        edge = Edge(
            from_node=from_node,
            to_node=to_node,
            distance_km=distance_km,
            free_flow_speed_kmh=free_flow_speed_kmh,
            travel_time_sec=travel_time_sec or 0.0,
            road_type=road_type,
        )
        self.adjacency[from_node].append(edge)

        if bidirectional:
            reverse_edge = Edge(
                from_node=to_node,
                to_node=from_node,
                distance_km=distance_km,
                free_flow_speed_kmh=free_flow_speed_kmh,
                travel_time_sec=travel_time_sec or 0.0,
                road_type=road_type,
            )
            self.adjacency[to_node].append(reverse_edge)

        return edge

    def get_neighbors(self, node_id: str) -> List[Edge]:
        return self.adjacency.get(node_id, [])

    def to_dict(self) -> Dict[str, Any]:
        edges_list = []
        for u, edge_list in self.adjacency.items():
            for e in edge_list:
                edges_list.append({
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "distance_km": e.distance_km,
                    "speed_kmh": e.free_flow_speed_kmh,
                    "travel_time_sec": e.travel_time_sec,
                    "road_type": e.road_type,
                })
        return {
            "name": self.name,
            "nodes": [
                {"node_id": n.node_id, "name": n.name, "x": n.x, "y": n.y, "zone": n.zone}
                for n in self.nodes.values()
            ],
            "edges": edges_list,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoadGraph:
        graph = cls(name=data.get("name", "RoadNetwork"))
        for n in data.get("nodes", []):
            graph.add_node(n["node_id"], n["name"], n.get("x", 0.0), n.get("y", 0.0), n.get("zone", "urban"))
        for e in data.get("edges", []):
            graph.add_edge(
                from_node=e["from_node"],
                to_node=e["to_node"],
                distance_km=e["distance_km"],
                free_flow_speed_kmh=e.get("speed_kmh", 30.0),
                travel_time_sec=e.get("travel_time_sec"),
                road_type=e.get("road_type", "arterial"),
                bidirectional=False,
            )
        return graph

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> RoadGraph:
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def create_small_test_graph() -> RoadGraph:
    """Creates a small 5-node test graph with known analytical shortest paths.

    Nodes: A, B, C, D, E
    Routes:
      A -> B (3 km, 30 km/h -> 360 s)
      B -> C (4 km, 30 km/h -> 480 s)
      A -> C (direct: 10 km, 30 km/h -> 1200 s, so A->B->C is 840 s < 1200 s)
      C -> D (2 km, 20 km/h -> 360 s)
      B -> D (direct: 8 km, 40 km/h -> 720 s)
      D -> E (5 km, 30 km/h -> 600 s)
      C -> E (6 km, 30 km/h -> 720 s)
    """
    g = RoadGraph(name="SmallValidationGraph")
    for nid in ["A", "B", "C", "D", "E"]:
        g.add_node(nid, f"Station_{nid}")

    g.add_edge("A", "B", distance_km=3.0, free_flow_speed_kmh=30.0, bidirectional=True)
    g.add_edge("B", "C", distance_km=4.0, free_flow_speed_kmh=30.0, bidirectional=True)
    g.add_edge("A", "C", distance_km=10.0, free_flow_speed_kmh=30.0, bidirectional=True)
    g.add_edge("C", "D", distance_km=2.0, free_flow_speed_kmh=20.0, bidirectional=True)
    g.add_edge("B", "D", distance_km=8.0, free_flow_speed_kmh=40.0, bidirectional=True)
    g.add_edge("D", "E", distance_km=5.0, free_flow_speed_kmh=30.0, bidirectional=True)
    g.add_edge("C", "E", distance_km=6.0, free_flow_speed_kmh=30.0, bidirectional=True)
    return g


def create_bengaluru_synthetic_network() -> RoadGraph:
    """Creates a realistic 15-node urban road network modeled on Bengaluru's core corridors.

    Key Hubs:
      N01: Majestic (Central Bus/Rail Terminus)
      N02: Malleshwaram (North-West Hub)
      N03: Indiranagar (East Commercial Hub)
      N04: Koramangala (South-East Hub)
      N05: HSR Layout (South-East Tech Hub)
      N06: Silk Board (Southern Arterial Junction)
      N07: Electronic City (Southern Tech Zone)
      N08: Jayanagar (South Residential Hub)
      N09: Banashankari (South-West Transit Junction)
      N10: Rajajinagar (West Hub)
      N11: Yeshwanthpur (North-West Rail/Bus)
      N12: Hebbal (North Junction / Airport Corridor)
      N13: Whitefield (East IT Corridor)
      N14: Marathahalli (East Ring Road Junction)
      N15: Bellandur (Outer Ring Road Corridor)
    """
    g = RoadGraph(name="BengaluruParatransitNetwork")

    nodes_data = [
        ("N01", "Majestic", 0.0, 0.0, "central"),
        ("N02", "Malleshwaram", -1.5, 3.0, "north_west"),
        ("N03", "Indiranagar", 6.0, 1.0, "east"),
        ("N04", "Koramangala", 5.0, -4.0, "south_east"),
        ("N05", "HSR_Layout", 7.0, -6.5, "south_east"),
        ("N06", "Silk_Board", 5.5, -7.0, "south_junction"),
        ("N07", "Electronic_City", 7.0, -15.0, "outer_south"),
        ("N08", "Jayanagar", 1.0, -5.0, "south"),
        ("N09", "Banashankari", -1.0, -8.0, "south_west"),
        ("N10", "Rajajinagar", -4.0, 1.5, "west"),
        ("N11", "Yeshwanthpur", -3.5, 6.0, "north_west"),
        ("N12", "Hebbal", 1.0, 8.0, "north"),
        ("N13", "Whitefield", 16.0, 2.0, "outer_east"),
        ("N14", "Marathahalli", 11.0, 0.0, "east_orr"),
        ("N15", "Bellandur", 9.5, -3.5, "east_orr"),
    ]

    for nid, name, x, y, zone in nodes_data:
        g.add_node(nid, name, x=x, y=y, zone=zone)

    # Road connections (distance_km, speed_kmh, road_type)
    corridors = [
        # Central Radial Corridors
        ("N01", "N02", 4.0, 28.0, "arterial"),
        ("N01", "N03", 6.5, 26.0, "arterial"),
        ("N01", "N04", 7.0, 25.0, "arterial"),
        ("N01", "N08", 5.5, 27.0, "arterial"),
        ("N01", "N10", 4.5, 28.0, "arterial"),

        # North & North-West
        ("N02", "N11", 3.5, 30.0, "arterial"),
        ("N10", "N11", 4.0, 32.0, "arterial"),
        ("N11", "N12", 5.5, 35.0, "arterial"),
        ("N01", "N12", 7.5, 32.0, "arterial"),

        # South & South-West
        ("N08", "N04", 4.0, 26.0, "arterial"),
        ("N08", "N09", 3.8, 28.0, "arterial"),
        ("N08", "N06", 4.5, 28.0, "arterial"),
        ("N09", "N06", 5.0, 28.0, "arterial"),

        # South-East & Electronic City Corridor
        ("N04", "N05", 3.2, 28.0, "arterial"),
        ("N04", "N06", 3.0, 25.0, "arterial"),
        ("N05", "N06", 2.2, 26.0, "arterial"),
        ("N06", "N07", 9.5, 42.0, "expressway"),

        # Outer Ring Road (ORR) Corridor (Silk Board -> HSR -> Bellandur -> Marathahalli)
        ("N06", "N15", 4.8, 30.0, "ring_road"),
        ("N05", "N15", 3.0, 32.0, "arterial"),
        ("N15", "N14", 4.2, 32.0, "ring_road"),
        ("N14", "N13", 6.5, 30.0, "arterial"),
        ("N03", "N14", 5.5, 30.0, "arterial"),
        ("N03", "N04", 4.8, 26.0, "arterial"),

        # Outer Ring North-East Link (Indiranagar / Marathahalli to Hebbal)
        ("N03", "N12", 8.5, 35.0, "ring_road"),
        ("N14", "N12", 13.0, 38.0, "ring_road"),
    ]

    for u, v, dist, speed, rtype in corridors:
        g.add_edge(u, v, distance_km=dist, free_flow_speed_kmh=speed, road_type=rtype, bidirectional=True)

    return g
