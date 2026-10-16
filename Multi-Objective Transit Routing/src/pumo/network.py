"""Compact shared multimodal network and timetable transition interface."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from pathlib import Path
from collections import defaultdict
from math import hypot

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from .model import Query
from .walking import WalkingGraph


@dataclass(frozen=True, slots=True)
class Transition:
    from_node: int
    to_node: int
    mode: str
    segment_id: str
    walk_time_s: int = 0
    distance_m: float = 0.0
    cost: float = 0.0
    emissions: float = 0.0
    risk: float = 0.0
    trip_id: str | None = None
    pattern_id: int | str | None = None
    departure_time: int | None = None
    arrival_time: int | None = None


class PreparedNetwork:
    """Adjacency and timetable shared by each algorithm.

    ``from_directory`` loads the cached network and optionally selects a circular
    region centered on the densest stop cluster. Explicitly constructed instances
    are convenient for toy correctness tests.
    """
    def __init__(self, coords=None, *, boarding_buffer_s: int = 30,
                 fare_per_boarding=None, emission_factors=None, edges=(),
                 coordinates_are_wgs84: bool = False):
        self.coords = coords or {}
        self.coordinates_are_wgs84 = coordinates_are_wgs84
        self.scenario = "toy"
        self.node_risk = {}
        self.boarding_buffer_s = int(boarding_buffer_s)
        self.fare_per_boarding = fare_per_boarding or {"BUS": 1.0, "METRO": 0.5}
        self.emission_factors = emission_factors or {"WALK": 0.0, "BUS": 1.0, "METRO": .5}
        self.walk: dict[int, list[Transition]] = defaultdict(list)
        self.transit: dict[int, list[Transition]] = defaultdict(list)
        self.stop_nodes: dict[int, int] = {}
        self.node_stops: dict[int, list[int]] = defaultdict(list)
        self._tree = None
        self._node_order = None
        for edge in edges:
            self.add(edge)

    def add(self, edge: Transition):
        if edge.mode == "WALK": self.walk[int(edge.from_node)].append(edge)
        else: self.transit[int(edge.from_node)].append(edge)
        self.coords.setdefault(int(edge.from_node), (0.0, 0.0))
        self.coords.setdefault(int(edge.to_node), (0.0, 0.0))

    def finalize(self):
        for node in self.transit:
            self.transit[node].sort(key=lambda e: (e.departure_time or 0, str(e.pattern_id), e.trip_id or ""))
        self._node_order = np.array(sorted(self.coords), dtype=np.int64)
        self._xy = np.array([self.coords[int(n)] for n in self._node_order], dtype=float)
        if self.coordinates_are_wgs84 and len(self._xy):
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
            self._xy = np.column_stack(transformer.transform(self._xy[:, 1], self._xy[:, 0]))
        self._tree = cKDTree(self._xy) if len(self._xy) else None
        return self

    def nearest_node(self, lat_lon: tuple[float, float]) -> int:
        if self._tree is None: self.finalize()
        if self._tree is None: raise ValueError("network has no nodes")
        lat, lon = lat_lon
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
        xy = transformer.transform(lon, lat)
        _, i = self._tree.query(xy)
        return int(self._node_order[int(i)])

    def transitions_from(self, node: int, current_time: int, on_trip: str | None,
                        *, boarding_buffer_s: int | None = None, disruptions=()):
        for edge in self.walk.get(int(node), ()):
            adjusted = _apply_disruption(edge, current_time, disruptions)
            if adjusted is not None: yield adjusted
        buffer_s = self.boarding_buffer_s if boarding_buffer_s is None else int(boarding_buffer_s)
        edges = self.transit.get(int(node), ())
        if not edges: return
        earliest: dict[object, Transition] = {}
        on_trip_edges = []
        for edge in edges:
            edge = _apply_disruption(edge, current_time, disruptions)
            if edge is None: continue
            dep = int(edge.departure_time)
            if edge.trip_id == on_trip and dep >= current_time:
                on_trip_edges.append(edge)
            threshold = current_time if edge.trip_id == on_trip else current_time + buffer_s
            if dep < threshold: continue
            key = edge.pattern_id if edge.pattern_id is not None else edge.trip_id
            if key not in earliest or dep < int(earliest[key].departure_time): earliest[key] = edge
        emitted = set()
        for edge in on_trip_edges + list(earliest.values()):
            key = (edge.trip_id, edge.to_node, edge.departure_time)
            if key not in emitted:
                emitted.add(key)
                yield edge

    @classmethod
    def from_directory(cls, directory: str | Path, *, scenario: str = "city",
                       radius_km: float | None = None, boarding_buffer_s: int = 30):
        """Load a scenario subgraph from prepared Parquet products."""
        p = Path(directory)
        nodes = gpd.read_parquet(p / "walking_nodes.parquet")
        edges = gpd.read_parquet(p / "walking_edges.parquet")
        stops = pd.read_parquet(p / "stops.parquet")
        connections = pd.read_parquet(p / "connections.parquet")
        trips = pd.read_parquet(p / "trips.parquet")
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
        node_xy = np.column_stack(transformer.transform(nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()))
        coords_all = {int(n): (float(x), float(y)) for n, (x, y) in zip(nodes.node_idx, node_xy)}
        radius_m = None if scenario.lower() == "city" else float(radius_km or (3 if scenario.lower() == "core" else 8)) * 1000
        stop_xy = np.column_stack(transformer.transform(stops.lon.to_numpy(), stops.lat.to_numpy()))
        if radius_m:
            neighborhoods = cKDTree(stop_xy).query_ball_point(stop_xy, r=radius_m)
            density = np.fromiter((len(group) for group in neighborhoods), dtype=np.int32, count=len(neighborhoods))
            center_idx = int(np.argmax(density))
            # A stop-density centroid is more stable than the first stop at the maximum.
            center = stop_xy[center_idx]
            keep = np.linalg.norm(node_xy - center, axis=1) <= radius_m
            kept_nodes = set(nodes.node_idx.to_numpy()[keep].astype(int))
        else:
            kept_nodes = set(nodes.node_idx.to_numpy().astype(int))
        coords = {n: coords_all[n] for n in kept_nodes}
        net = cls(coords, boarding_buffer_s=boarding_buffer_s)
        net.scenario = scenario.lower()
        edges = edges[edges.from_node.isin(kept_nodes) & edges.to_node.isin(kept_nodes)]
        node_risk_values=defaultdict(list)
        for row in edges.itertuples(index=False):
            net.add(Transition(int(row.from_node), int(row.to_node), "WALK", str(row.edge_id),
                int(row.walk_time_s), float(row.length_m), risk=float(row.length_m * (row.safety_risk if pd.notna(row.safety_risk) else 0.0))))
            node_risk_values[int(row.from_node)].append(float(row.safety_risk) if pd.notna(row.safety_risk) else 0.0)
        net.node_risk={n:float(np.mean(values)) for n,values in node_risk_values.items()}
        stops = stops[stops.node_idx.isin(kept_nodes)].copy()
        for row in stops.itertuples(index=False):
            net.stop_nodes[int(row.stop_idx)] = int(row.node_idx)
            net.node_stops[int(row.node_idx)].append(int(row.stop_idx))
        st = connections[connections.from_stop.isin(stops.stop_idx) & connections.to_stop.isin(stops.stop_idx)].copy()
        st = st.merge(stops[["stop_idx", "node_idx", "lat", "lon"]].rename(columns={"stop_idx":"from_stop", "node_idx":"from_node", "lat":"from_lat", "lon":"from_lon"}), on="from_stop")
        st = st.merge(stops[["stop_idx", "node_idx", "lat", "lon"]].rename(columns={"stop_idx":"to_stop", "node_idx":"to_node", "lat":"to_lat", "lon":"to_lon"}), on="to_stop")
        st = st.merge(trips[["trip_idx", "trip_id", "pattern_idx", "mode"]].drop_duplicates("trip_idx"), on="trip_idx", suffixes=("", "_trip"))
        start_xy = np.column_stack(transformer.transform(st.from_lon.to_numpy(), st.from_lat.to_numpy()))
        end_xy = np.column_stack(transformer.transform(st.to_lon.to_numpy(), st.to_lat.to_numpy()))
        lengths = np.linalg.norm(end_xy - start_xy, axis=1)
        for i, row in enumerate(st.itertuples(index=False)):
            mode = str(row.mode)
            base_cost = .1 if mode == "METRO" else 0.0
            net.add(Transition(int(row.from_node), int(row.to_node), mode,
                f"{row.trip_id}:{row.from_stop}>{row.to_stop}", distance_m=float(lengths[i]),
                cost=base_cost, emissions=float(lengths[i] / 1000 * net.emission_factors[mode]),
                trip_id=str(row.trip_id), pattern_id=f"{mode}:{row.pattern_idx}",
                departure_time=int(row.dep_time), arrival_time=int(row.arr_time)))
        net.finalize()
        return net


def _apply_disruption(edge: Transition, current_time: int, disruptions):
    """Apply immutable synthetic event overlays to an individual transition."""
    adjusted = edge
    for event in disruptions:
        t = int(edge.departure_time) if edge.departure_time is not None else int(current_time)
        active = int(event.start_time) <= t < int(event.end_time)
        if not active: continue
        ids = {str(i) for i in getattr(event, "edge_ids", ())}
        if edge.mode == "WALK" and edge.segment_id in ids and getattr(event, "closure", False):
            return None
        if edge.mode != "WALK" and edge.trip_id in set(getattr(event, "trip_ids", ())) and \
                getattr(event, "event_type", "") in {"BUS_DELAY", "METRO_DELAY"}:
            penalty = int(getattr(event, "time_penalty_s", 0))
            adjusted = replace(adjusted, departure_time=int(adjusted.departure_time)+penalty,
                               arrival_time=int(adjusted.arrival_time)+penalty)
        if edge.mode == "WALK" and edge.segment_id in ids and getattr(event, "event_type", "") == "LIGHTING_OUTAGE":
            adjusted = replace(adjusted, risk=adjusted.risk + float(getattr(event, "safety_penalty", 0.0)))
    return adjusted
