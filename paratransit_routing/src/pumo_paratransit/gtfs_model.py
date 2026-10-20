"""
Paper 1 — Digital Matatu (Williams et al. 2015)
------------------------------------------------
GTFS-compliant data structures for shared-auto routes and stops.

What we take from Williams et al. (2015):
  - Modified GTFS format for semi-formal paratransit routes
  - Route geometry (shape) and stop locations as first-class entities
  - A data validation pipeline that flags missing or inconsistent entries
  - Network representation suitable for connecting stops to the road graph

Reference:
  Williams S. et al. (2015). "The Digital Matatu Project: Using Cell Phones
  to Create an Open Source Data for Nairobi's Semi-Formal Bus System."
  Journal of Transport Geography, 49, 39-51.
  https://doi.org/10.1016/j.jtrangeo.2015.10.005
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core GTFS-like entities
# ---------------------------------------------------------------------------

@dataclass
class AutoStop:
    """A designated pickup/drop-off point for a shared-auto route.

    Analogous to a GTFS 'stop', extended with shared-auto-specific fields:
      - zone_type: whether the stop is a formal terminus, corridor stop, or informal hail point
      - road_node_id: links the stop to a node in the RoadGraph for routing
    """
    stop_id: str
    name: str
    latitude: float = 0.0
    longitude: float = 0.0
    road_node_id: Optional[str] = None          # Link to RoadGraph node
    zone_type: str = "corridor"                  # "terminus" | "corridor" | "informal"
    has_shelter: bool = False
    avg_pax_per_hour: float = 0.0               # Demand proxy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_id": self.stop_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "road_node_id": self.road_node_id,
            "zone_type": self.zone_type,
            "has_shelter": self.has_shelter,
            "avg_pax_per_hour": self.avg_pax_per_hour,
        }


@dataclass
class StopTime:
    """Scheduled or approximate arrival of an auto at a stop on a trip.

    Shared autos have highly variable headways, so 'arrival_offset_sec' is
    an expected travel time from route origin rather than a fixed clock time.
    """
    stop_id: str
    stop_sequence: int
    arrival_offset_sec: float       # seconds from route start
    departure_offset_sec: float     # seconds from route start (usually arrival + service_time)
    pickup_type: int = 0            # 0 = regular, 1 = no pickup, 3 = phone-ahead
    drop_off_type: int = 0          # 0 = regular, 1 = no drop-off


@dataclass
class RouteShape:
    """Sequence of (lat, lon) waypoints describing the physical road path.

    Williams et al. 2015 collected these via phone GPS traces during survey rides.
    """
    shape_id: str
    points: List[Tuple[float, float]] = field(default_factory=list)  # (lat, lon)

    def add_point(self, lat: float, lon: float) -> None:
        self.points.append((lat, lon))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape_id": self.shape_id,
            "points": [{"lat": p[0], "lon": p[1]} for p in self.points],
        }


@dataclass
class SharedAutoRoute:
    """A named shared-auto corridor with ordered stops, shape, and metadata.

    Analogous to a GTFS 'route' + 'trip' combined.  Shared autos typically
    operate fixed corridors with flexible intermediate hailing rather than
    purely fixed schedules.
    """
    route_id: str
    route_name: str
    direction: str = "outbound"                  # "outbound" | "inbound"
    vehicle_capacity: int = 3                    # Standard auto-rickshaw: 3 passengers
    stop_times: List[StopTime] = field(default_factory=list)
    shape: Optional[RouteShape] = None
    operator: str = "informal"                   # "informal" | "formal" | "aggregated"
    active: bool = True

    def add_stop(self, stop_id: str, arrival_offset_sec: float, service_duration_sec: float = 30.0) -> None:
        seq = len(self.stop_times)
        self.stop_times.append(StopTime(
            stop_id=stop_id,
            stop_sequence=seq,
            arrival_offset_sec=arrival_offset_sec,
            departure_offset_sec=arrival_offset_sec + service_duration_sec,
        ))

    def ordered_stop_ids(self) -> List[str]:
        return [st.stop_id for st in sorted(self.stop_times, key=lambda s: s.stop_sequence)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "route_name": self.route_name,
            "direction": self.direction,
            "vehicle_capacity": self.vehicle_capacity,
            "operator": self.operator,
            "active": self.active,
            "stop_sequence": [
                {
                    "stop_id": st.stop_id,
                    "seq": st.stop_sequence,
                    "arr_offset_sec": st.arrival_offset_sec,
                    "dep_offset_sec": st.departure_offset_sec,
                }
                for st in sorted(self.stop_times, key=lambda s: s.stop_sequence)
            ],
            "shape": self.shape.to_dict() if self.shape else None,
        }


# ---------------------------------------------------------------------------
# Route Registry: multi-route container with validation
# ---------------------------------------------------------------------------

class SharedAutoNetwork:
    """Registry of all shared-auto routes and stops for a city area.

    Implements the Digital Matatu data model:
      - Stores stops and routes in indexed dictionaries
      - Validates referential integrity (every stop_id in a route must exist)
      - Flags missing road_node_id links to the RoadGraph
      - Exports / imports from JSON
    """

    def __init__(self, city: str = "Bengaluru"):
        self.city = city
        self.stops: Dict[str, AutoStop] = {}
        self.routes: Dict[str, SharedAutoRoute] = {}

    def add_stop(self, stop: AutoStop) -> None:
        self.stops[stop.stop_id] = stop

    def add_route(self, route: SharedAutoRoute) -> None:
        self.routes[route.route_id] = route

    def validate(self) -> List[str]:
        """Returns a list of validation error messages.

        Checks:
          1. Every stop_id referenced in a route exists in self.stops.
          2. Every stop used in routing has a road_node_id linking it to the graph.
          3. Routes have at least 2 stops (origin and destination).
        """
        errors: List[str] = []
        for route_id, route in self.routes.items():
            stop_ids = route.ordered_stop_ids()
            if len(stop_ids) < 2:
                errors.append(f"Route {route_id}: fewer than 2 stops (origin and destination required)")
            for sid in stop_ids:
                if sid not in self.stops:
                    errors.append(f"Route {route_id}: references unknown stop_id '{sid}'")
                elif self.stops[sid].road_node_id is None:
                    errors.append(
                        f"Stop '{sid}' (used in route {route_id}) has no road_node_id — "
                        f"cannot connect to road graph for routing"
                    )
        return errors

    def get_road_nodes_for_route(self, route_id: str) -> List[str]:
        """Returns the ordered road graph node IDs for a route's stops."""
        route = self.routes[route_id]
        nodes = []
        for sid in route.ordered_stop_ids():
            stop = self.stops.get(sid)
            if stop and stop.road_node_id:
                nodes.append(stop.road_node_id)
        return nodes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "city": self.city,
            "stops": [s.to_dict() for s in self.stops.values()],
            "routes": [r.to_dict() for r in self.routes.values()],
        }

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved SharedAutoNetwork to {filepath}")

    @classmethod
    def load_json(cls, filepath: str) -> SharedAutoNetwork:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        net = cls(city=data.get("city", "Unknown"))
        for s in data.get("stops", []):
            net.add_stop(AutoStop(**{k: v for k, v in s.items()}))
        for r in data.get("routes", []):
            route = SharedAutoRoute(
                route_id=r["route_id"],
                route_name=r["route_name"],
                direction=r.get("direction", "outbound"),
                vehicle_capacity=r.get("vehicle_capacity", 3),
                operator=r.get("operator", "informal"),
                active=r.get("active", True),
            )
            for st in r.get("stop_sequence", []):
                route.stop_times.append(StopTime(
                    stop_id=st["stop_id"],
                    stop_sequence=st["seq"],
                    arrival_offset_sec=st["arr_offset_sec"],
                    departure_offset_sec=st["dep_offset_sec"],
                ))
            if r.get("shape"):
                shape = RouteShape(shape_id=r["shape"]["shape_id"])
                for pt in r["shape"]["points"]:
                    shape.add_point(pt["lat"], pt["lon"])
                route.shape = shape
            net.add_route(route)
        return net


# ---------------------------------------------------------------------------
# Bengaluru Synthetic Shared-Auto Network (modeled on real corridors)
# ---------------------------------------------------------------------------

def create_bengaluru_shared_auto_network() -> SharedAutoNetwork:
    """Creates a synthetic shared-auto network modeled on Bengaluru's major corridors.

    Based on Digital Matatu methodology: routes are organized along major
    arterials, with stops at major intersections and activity hubs.
    Each stop is linked to the corresponding road graph node.
    """
    net = SharedAutoNetwork(city="Bengaluru")

    # Stops mapped to road network nodes
    stop_data = [
        ("S01", "Majestic Bus Stand", 12.9780, 77.5720, "N01", "terminus"),
        ("S02", "Malleshwaram 18th Cross", 13.0070, 77.5700, "N02", "corridor"),
        ("S03", "Indiranagar 100ft Road", 12.9784, 77.6408, "N03", "terminus"),
        ("S04", "Koramangala 5th Block", 12.9356, 77.6278, "N04", "terminus"),
        ("S05", "HSR Layout Sector 2", 12.9116, 77.6412, "N05", "corridor"),
        ("S06", "Silk Board Junction", 12.9175, 77.6227, "N06", "terminus"),
        ("S07", "Electronic City Phase 1", 12.8399, 77.6770, "N07", "terminus"),
        ("S08", "Jayanagar 4th Block", 12.9300, 77.5820, "N08", "corridor"),
        ("S09", "Banashankari Temple", 12.9258, 77.5490, "N09", "corridor"),
        ("S10", "Rajajinagar Circle", 12.9921, 77.5505, "N10", "corridor"),
        ("S11", "Yeshwanthpur Circle", 13.0250, 77.5410, "N11", "terminus"),
        ("S12", "Hebbal Flyover", 13.0452, 77.5972, "N12", "terminus"),
        ("S13", "Whitefield ITPL Gate", 12.9718, 77.7500, "N13", "terminus"),
        ("S14", "Marathahalli Bridge", 12.9565, 77.7011, "N14", "corridor"),
        ("S15", "Bellandur Lake Road", 12.9273, 77.6726, "N15", "corridor"),
    ]

    for sid, name, lat, lon, node_id, zone in stop_data:
        net.add_stop(AutoStop(
            stop_id=sid, name=name, latitude=lat, longitude=lon,
            road_node_id=node_id, zone_type=zone
        ))

    # Route R01: Majestic → Malleshwaram → Yeshwanthpur → Hebbal (North Corridor)
    r01 = SharedAutoRoute("R01", "Majestic - Hebbal North Corridor", vehicle_capacity=3)
    for sid, offset in [("S01", 0), ("S02", 600), ("S10", 900), ("S11", 1320), ("S12", 2220)]:
        r01.add_stop(sid, offset)
    net.add_route(r01)

    # Route R02: Majestic → Jayanagar → Banashankari (South Corridor)
    r02 = SharedAutoRoute("R02", "Majestic - Banashankari South Corridor", vehicle_capacity=3)
    for sid, offset in [("S01", 0), ("S08", 900), ("S09", 1380)]:
        r02.add_stop(sid, offset)
    net.add_route(r02)

    # Route R03: Indiranagar → Koramangala → HSR → Silk Board (East-South Link)
    r03 = SharedAutoRoute("R03", "Indiranagar - Silk Board Link", vehicle_capacity=3)
    for sid, offset in [("S03", 0), ("S04", 1080), ("S05", 1680), ("S06", 2400)]:
        r03.add_stop(sid, offset)
    net.add_route(r03)

    # Route R04: Silk Board → Electronic City (Expressway Feeder)
    r04 = SharedAutoRoute("R04", "Silk Board - Electronic City Expressway", vehicle_capacity=4)
    for sid, offset in [("S06", 0), ("S07", 1200)]:
        r04.add_stop(sid, offset)
    net.add_route(r04)

    # Route R05: Outer Ring Road (Indiranagar → Marathahalli → Bellandur → HSR)
    r05 = SharedAutoRoute("R05", "ORR East Corridor", vehicle_capacity=3)
    for sid, offset in [("S03", 0), ("S14", 1200), ("S15", 1800), ("S05", 2400)]:
        r05.add_stop(sid, offset)
    net.add_route(r05)

    return net
