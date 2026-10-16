#!/usr/bin/env python3
"""Build a cached shared walking/transit/safety network from Data/.

Raw source files are opened read-only. All products go under derived_data/ because
Windows treats the raw ``Data`` and generated ``data`` names as the same path.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import Point, box
from pyrosm import OSM
import yaml
from pumo.timetable import gtfs_time_seconds

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data"
OUT = ROOT / "derived_data" / "processed"
METRIC_CRS = "EPSG:32643"
REGION = (77.35, 12.75, 77.85, 13.25)
HIGHWAYS = {"footway", "pedestrian", "path", "steps", "living_street", "residential",
            "service", "unclassified", "tertiary", "secondary", "primary", "crossing"}


def read_config():
    with (ROOT / "config" / "thresholds.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def gtfs_time(value: str) -> int:
    """GTFS time in seconds after midnight; hours may exceed 23."""
    return gtfs_time_seconds(value)


def pick_service_date(feed_dirs: list[Path], explicit: str | None) -> str:
    if explicit:
        return explicit.replace("-", "")
    lo, hi = date.min, date.max
    for feed in feed_dirs:
        cal = pd.read_csv(feed / "calendar.txt", dtype=str)
        starts = pd.to_datetime(cal["start_date"], format="%Y%m%d", errors="coerce")
        ends = pd.to_datetime(cal["end_date"], format="%Y%m%d", errors="coerce")
        if starts.notna().any(): lo = max(lo, starts.min().date())
        if ends.notna().any(): hi = min(hi, ends.max().date())
    d = lo
    while d <= hi and d.weekday() >= 5:
        d += timedelta(days=1)
    if d > hi:
        raise ValueError("BMTC and BMRCL calendars have no overlapping weekday")
    return d.strftime("%Y%m%d")


def active_service_ids(feed: Path, service_date: str) -> set[str]:
    day = date(int(service_date[:4]), int(service_date[4:6]), int(service_date[6:8]))
    cal_path = feed / "calendar.txt"
    active: set[str] = set()
    if cal_path.exists():
        cal = pd.read_csv(cal_path, dtype=str).fillna("")
        wd = day.strftime("%A").lower()
        for row in cal.to_dict("records"):
            if row.get("start_date", "") <= service_date <= row.get("end_date", "") and row.get(wd) == "1":
                active.add(row["service_id"])
    exceptions = feed / "calendar_dates.txt"
    if exceptions.exists():
        ex = pd.read_csv(exceptions, dtype=str).fillna("")
        for row in ex.loc[ex["date"] == service_date].to_dict("records"):
            if row["exception_type"] == "1": active.add(row["service_id"])
            elif row["exception_type"] == "2": active.discard(row["service_id"])
    return active


def parse_feed(feed: Path, mode: str, service_date: str, region_bbox) -> tuple[pd.DataFrame, ...]:
    stops = pd.read_csv(feed / "stops.txt", dtype={"stop_id": str}, low_memory=False)
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    valid_stops = stops[stops.stop_lat.between(region_bbox[1], region_bbox[3]) &
                        stops.stop_lon.between(region_bbox[0], region_bbox[2])].copy()
    invalid_stops = stops.loc[~stops.index.isin(valid_stops.index)].copy()
    if not invalid_stops.empty:
        invalid_stops["quarantine_reason"] = "stop coordinates outside Bengaluru clip or invalid"
    routes = pd.read_csv(feed / "routes.txt", dtype=str, low_memory=False).fillna("")
    trips = pd.read_csv(feed / "trips.txt", dtype=str, low_memory=False).fillna("")
    services = active_service_ids(feed, service_date)
    trips = trips[trips.service_id.isin(services)].copy()
    raw_st = pd.read_csv(feed / "stop_times.txt", dtype={"trip_id": str, "stop_id": str}, low_memory=False)
    raw_st = raw_st[raw_st.trip_id.isin(set(trips.trip_id))].copy()
    raw_st["stop_sequence"] = pd.to_numeric(raw_st["stop_sequence"], errors="coerce")
    raw_st["arrival_sec"] = raw_st.arrival_time.map(gtfs_time)
    raw_st["departure_sec"] = raw_st.departure_time.map(gtfs_time)
    raw_st = raw_st.sort_values(["trip_id", "stop_sequence"], kind="stable")
    invalid_trip_ids = set()
    for trip_id, group in raw_st.groupby("trip_id", sort=False):
        arr, dep = group.arrival_sec.to_numpy(), group.departure_sec.to_numpy()
        if np.any(arr > dep) or np.any(np.diff(arr) <= 0) or np.any(np.diff(dep) < 0):
            invalid_trip_ids.add(trip_id)
    if invalid_trip_ids:
        bad = raw_st[raw_st.trip_id.isin(invalid_trip_ids)].copy()
        bad["quarantine_reason"] = "non-monotone or invalid trip stop times"
        qdir = ROOT / "derived_data" / "quarantine"
        qdir.mkdir(parents=True, exist_ok=True)
        bad.to_csv(qdir / f"{mode.lower()}_invalid_stop_times.csv", index=False)
        trips = trips[~trips.trip_id.isin(invalid_trip_ids)]
        raw_st = raw_st[~raw_st.trip_id.isin(invalid_trip_ids)]
    stop_ids = set(valid_stops.stop_id)
    raw_st = raw_st[raw_st.stop_id.isin(stop_ids)].copy()
    trip_ids_all_stops = raw_st.groupby("trip_id").stop_id.nunique()
    # Keep only trips whose entire scheduled sequence survived coordinate validation.
    original_counts = pd.read_csv(feed / "stop_times.txt", usecols=["trip_id"], dtype=str).trip_id.value_counts()
    valid_trip_ids = set(trip_ids_all_stops.index[trip_ids_all_stops == trip_ids_all_stops.index.map(original_counts).to_numpy()])
    trips = trips[trips.trip_id.isin(valid_trip_ids)].copy()
    raw_st = raw_st[raw_st.trip_id.isin(valid_trip_ids)].copy()
    route_by_id = routes.set_index("route_id", drop=False)
    trip_routes = trips[["trip_id", "route_id", "service_id"]].copy()
    trip_routes["agency"] = mode
    trip_routes = trip_routes.join(routes.set_index("route_id")[[c for c in ("route_short_name", "route_long_name") if c in routes]], on="route_id")
    raw_st = raw_st.merge(trip_routes[["trip_id", "route_id", "service_id"]], on="trip_id", how="inner")
    raw_st["agency"] = mode
    # Use stable integer IDs internally while preserving the feed IDs.
    stop_map = {sid: i for i, sid in enumerate(valid_stops.stop_id.astype(str))}
    trip_map = {tid: i for i, tid in enumerate(trips.trip_id.astype(str))}
    route_map = {rid: i for i, rid in enumerate(routes.route_id.astype(str))}
    valid_stops["stop_idx"] = valid_stops.stop_id.astype(str).map(stop_map)
    valid_stops["agency"] = mode
    valid_stops["mode"] = mode
    valid_stops["lat"] = valid_stops.stop_lat
    valid_stops["lon"] = valid_stops.stop_lon
    trips["trip_idx"] = trips.trip_id.astype(str).map(trip_map)
    trips["route_idx"] = trips.route_id.astype(str).map(route_map)
    trips["agency"] = mode
    trips["mode"] = mode
    raw_st["trip_idx"] = raw_st.trip_id.astype(str).map(trip_map)
    raw_st["route_idx"] = raw_st.route_id.astype(str).map(route_map)
    raw_st["stop_idx"] = raw_st.stop_id.astype(str).map(stop_map)
    raw_st["mode"] = mode
    conns = raw_st.sort_values(["trip_idx", "stop_sequence"], kind="stable").copy()
    conns["from_stop"] = conns.groupby("trip_idx").stop_idx.shift()
    conns["dep_time"] = conns.groupby("trip_idx").departure_sec.shift()
    conns["to_stop"] = conns.stop_idx
    conns["arr_time"] = conns.arrival_sec
    conns = conns[conns.from_stop.notna()].copy()
    trip_groups = {tid: group.sort_values("stop_sequence", kind="stable")
                   for tid, group in raw_st.groupby("trip_id", sort=False)}
    by_stop_pattern = defaultdict(list)
    for tid, group in trip_groups.items(): by_stop_pattern[tuple(group.stop_id.astype(str))].append(tid)
    trip_pattern = {}; next_pattern_id = 0
    for stop_pattern, trip_ids in by_stop_pattern.items():
        trip_ids.sort(key=lambda tid: int(trip_groups[tid].departure_sec.iloc[0]))
        clusters = []
        for tid in trip_ids:
            arrivals = trip_groups[tid].arrival_sec.to_numpy(dtype=np.int64)
            departures = trip_groups[tid].departure_sec.to_numpy(dtype=np.int64)
            assigned = False
            for cluster in clusters:
                if len(arrivals) == len(cluster["last_arrivals"]) and np.all(arrivals >= cluster["last_arrivals"]):
                    cluster["last_arrivals"] = arrivals
                    trip_pattern[tid] = cluster["pattern_id"]
                    assigned = True
                    break
            if not assigned:
                clusters.append({"pattern_id": next_pattern_id, "last_arrivals": arrivals})
                trip_pattern[tid] = next_pattern_id
                next_pattern_id += 1
    trips["pattern_idx"] = trips.trip_id.map(trip_pattern)
    return valid_stops, routes, trips, raw_st, conns, invalid_stops


def build_walking_graph(pbf: Path, cfg: dict, clip_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict]:
    west, south, east, north = cfg["city_bbox"]
    buffer_deg = cfg["clip_buffer_km"] / 100.0  # conservative approx around 13N; metric buffer is applied below
    clip = box(west-buffer_deg, south-buffer_deg, east+buffer_deg, north+buffer_deg)
    # Ensure pyrosm's temporary shards remain in the authorized project output tree.
    tmp = ROOT / "derived_data" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    osm = OSM(str(pbf), bounding_box=clip, keep_node_info=True, engine="out_of_core")
    nodes, edges = osm.get_network(network_type="walking", nodes=True,
        tags_to_keep=["highway", "foot", "access", "oneway", "oneway:foot", "lit", "crossing", "street_lamp"])
    if nodes is None or edges is None or edges.empty:
        raise RuntimeError("OSM PBF parsed but no walkable edges were found in configured clip")
    edges = edges[edges.highway.astype(str).isin(HIGHWAYS)].copy()
    if "foot" in edges: edges = edges[~edges.foot.astype(str).str.lower().eq("no")]
    if "access" in edges: edges = edges[~edges.access.astype(str).str.lower().eq("no")]
    projected_lengths = gpd.GeoSeries(edges.geometry, crs="EPSG:4326").to_crs(METRIC_CRS).length
    edges = edges.loc[projected_lengths > 0].copy()
    nodes = nodes[nodes.id.isin(set(edges.u) | set(edges.v))].copy()
    # Keep the largest undirected component.
    ids = nodes.id.astype(np.int64).to_numpy()
    index = {int(v): i for i, v in enumerate(ids)}
    us = edges.u.map(index).to_numpy(); vs = edges.v.map(index).to_numpy()
    graph = coo_matrix((np.ones(len(us)*2), (np.r_[us,vs], np.r_[vs,us])), shape=(len(ids), len(ids))).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    main_component = int(np.argmax(sizes))
    keep_ids = set(ids[labels == main_component].tolist())
    nodes = nodes[nodes.id.isin(keep_ids)].copy()
    edges = edges[edges.u.isin(keep_ids) & edges.v.isin(keep_ids)].copy()
    nodes["in_city"] = nodes.geometry.apply(lambda g: west <= g.x <= east and south <= g.y <= north)
    # Expand each OSM segment to directed pedestrian arcs; ordinary road one-way does not restrict walking.
    onefoot = edges.get("oneway:foot", pd.Series("", index=edges.index)).astype(str).str.lower().isin({"yes", "1", "true"})
    directed = edges.copy()
    reverse = edges.loc[~onefoot].copy()
    reverse[["u", "v"]] = reverse[["v", "u"]].to_numpy()
    reverse["geometry"] = reverse.geometry.apply(lambda g: g.reverse())
    directed = pd.concat([directed, reverse], ignore_index=True)
    directed["edge_id"] = np.arange(len(directed), dtype=np.int64)
    directed["from_node"] = directed.u.astype(np.int64)
    directed["to_node"] = directed.v.astype(np.int64)
    directed["length_m"] = gpd.GeoSeries(directed.geometry, crs="EPSG:4326").to_crs(METRIC_CRS).length.to_numpy()
    directed["osm_way_id"] = directed.id.astype(np.int64)
    directed["road_class"] = directed.highway.astype(str)
    directed["walk_time_s"] = np.ceil(directed.length_m / float(cfg["walk_speed_mps"])).astype(np.int32)
    directed["base_time_sec"] = directed.walk_time_s
    directed["in_city"] = directed.geometry.apply(lambda g: west <= g.centroid.x <= east and south <= g.centroid.y <= north)
    nodes = nodes[["id", "lon", "lat", "in_city", "geometry"]].rename(columns={"id": "node_idx", "lon": "lon", "lat": "lat"})
    nodes["osm_node_id"] = nodes.node_idx
    nodes["node_idx"] = pd.factorize(nodes.osm_node_id, sort=True)[0].astype(np.int64)
    osm_to_idx = dict(zip(nodes.osm_node_id.astype(int), nodes.node_idx.astype(int)))
    directed["osm_from_node"] = directed.from_node
    directed["osm_to_node"] = directed.to_node
    directed["from_node"] = directed.osm_from_node.map(osm_to_idx).astype(np.int64)
    directed["to_node"] = directed.osm_to_node.map(osm_to_idx).astype(np.int64)
    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")
    directed = gpd.GeoDataFrame(directed.drop(columns=["u", "v"]), geometry="geometry", crs="EPSG:4326")
    nodes.to_parquet(clip_path.parent / "walking_nodes.parquet", index=False)
    directed.to_parquet(clip_path, index=False)
    coverage = {"source_pbf": pbf.name, "bbox": [west, south, east, north],
        "buffer_km": cfg["clip_buffer_km"], "nodes": int(len(nodes)), "directed_edges": int(len(directed)),
        "undirected_largest_component_share": float(sizes[main_component] / max(1, len(ids))),
        "components_before_lcc": int(n_components), "largest_component_nodes": int(sizes[main_component])}
    return nodes, directed, coverage


def add_safety(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, cfg: dict) -> tuple[gpd.GeoDataFrame, dict]:
    west, south, east, north = cfg["city_bbox"]
    transformer = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    edges_m = edges.to_crs(METRIC_CRS)
    safety_components = []
    coverage = {}
    # Road design proxy: normalized road-class ordinal, explicitly MODELED.
    rank = {"footway": .1, "pedestrian": .05, "path": .1, "steps": .15, "living_street": .25,
            "residential": .4, "service": .35, "unclassified": .5, "tertiary": .6,
            "secondary": .75, "primary": .9, "crossing": .45}
    edges["road_risk"] = edges.road_class.map(rank).fillna(.5).astype(float)
    edges["road_risk_source_type"] = "MODELED"
    safety_components.append("road_risk")
    coverage["road_risk"] = {"available_edges": int(len(edges)), "source_type": "MODELED"}
    # Flood KML contains point locations, not polygons. Any mapped point within radius contributes exposure.
    kml = RAW / "floodZone data.kml"
    flood_points = []
    if kml.exists():
        ns = {"k": "http://www.opengis.net/kml/2.2"}
        root = ET.parse(kml).getroot()
        for element in root.findall(".//k:Point/k:coordinates", ns):
            xy = (element.text or "").strip().split(",")
            if len(xy) >= 2:
                flood_points.append(Point(float(xy[0]), float(xy[1])))
    if flood_points:
        flood = gpd.GeoDataFrame(geometry=flood_points, crs="EPSG:4326").to_crs(METRIC_CRS)
        tree = flood.sindex
        r = float(cfg["flood_join_radius_m"])
        vals = []
        for geom in edges_m.geometry:
            candidates = list(tree.query(geom.buffer(r)))
            vals.append(1.0 if any(geom.distance(flood.geometry.iloc[i]) <= r for i in candidates) else 0.0)
        edges["flood_risk"] = vals
        edges["flood_risk_source_type"] = "DERIVED"
        safety_components.append("flood_risk")
        coverage["flood_risk"] = {"features": len(flood_points), "available_edges": int(sum(vals)),
                                    "source_type": "DERIVED", "join_radius_m": r}
    else:
        edges["flood_risk"] = np.nan
        coverage["flood_risk"] = {"status": "skipped", "reason": "no flood geometries"}
    # Ward-level streetlight totals are contextual only; use OSM lamp tags if present.
    lamps = []
    raw_lamp = RAW / "streetlight data.csv"
    coverage["streetlight_csv"] = {"status": "context_only", "reason": "ward counts have no coordinates"}
    if "street_lamp" in edges.columns:
        lamps = list(edges.loc[edges.street_lamp.notna(), "geometry"].apply(lambda g: g.interpolate(.5, normalized=True)))
    if lamps:
        lamp_gdf = gpd.GeoDataFrame(geometry=lamps, crs="EPSG:4326").to_crs(METRIC_CRS)
        lamp_tree = lamp_gdf.sindex
        radius = 25.0
        lit = []
        for geom in edges_m.geometry:
            candidates = list(lamp_tree.query(geom.buffer(radius)))
            lit.append(any(geom.distance(lamp_gdf.geometry.iloc[i]) <= radius for i in candidates))
        edges["lighting_risk"] = [0.0 if x else np.nan for x in lit]
        known = edges.lighting_risk.notna()
        median = float(edges.loc[known, "lighting_risk"].median()) if known.any() else .5
        edges["lighting_imputed"] = ~known
        edges["lighting_risk"] = edges.lighting_risk.fillna(median)
        safety_components.append("lighting_risk")
        coverage["lighting_risk"] = {"mapped_lamp_features": len(lamps), "edge_lamp_coverage": float(known.mean()),
            "imputed_edges": int((~known).sum()), "source_type": "OSM; imputation MEDIAN"}
    else:
        edges["lighting_risk"] = np.nan
        edges["lighting_imputed"] = True
        coverage["lighting_risk"] = {"status": "skipped", "reason": "no geolocated OSM street lamps found"}
    # Stop risk is assigned after GTFS snapping; this proxy remains separately recorded.
    weight_doc = yaml.safe_load((ROOT / "config" / "safety_weights.yaml").read_text(encoding="utf-8"))
    source_weight = weight_doc.get("weights", {})
    component_weight = {"road_risk": float(source_weight.get("road", 0.0)),
                        "flood_risk": float(source_weight.get("flood", 0.0)),
                        "lighting_risk": float(source_weight.get("lighting", 0.0))}
    numerator = sum(edges[c].fillna(0.0) * component_weight[c] for c in safety_components)
    denominator = sum(edges[c].notna().astype(float) * component_weight[c] for c in safety_components)
    edges["safety_risk"] = numerator / denominator.replace(0, np.nan)
    edges["component_flags"] = edges.apply(lambda row: ",".join(c for c in safety_components if pd.notna(row[c])), axis=1)
    edges["s_imputed"] = edges.component_flags.str.split(",").str.len() < len(safety_components)
    coverage["safety_components"] = safety_components
    coverage["s_imputed_edge_share"] = float(edges.s_imputed.mean())
    return edges, coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-date", help="GTFS service date, YYYYMMDD")
    parser.add_argument("--skip-osm", action="store_true", help="Reuse cached OSM products")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = read_config()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "derived_data" / "quarantine").mkdir(parents=True, exist_ok=True)
    feeds = [RAW / "bmtc", RAW / "bmrcl"]
    service_date = pick_service_date(feeds, args.service_date)
    cache_edges = OUT / "walking_edges.parquet"
    cache_nodes = OUT / "walking_nodes.parquet"
    stats = {"service_date": service_date, "crs_storage": "EPSG:4326", "metric_crs": METRIC_CRS,
             "provenance_policy": "MEASURED inputs; DERIVED joins; MODELED road proxy and walking times; PLACEHOLDER factors"}
    if args.skip_osm:
        if not cache_edges.exists() or not cache_nodes.exists(): raise FileNotFoundError("cached walking graph missing")
        nodes = gpd.read_parquet(cache_nodes); edges = gpd.read_parquet(cache_edges)
    else:
        if not args.force and cache_edges.exists() and cache_nodes.exists():
            nodes = gpd.read_parquet(cache_nodes); edges = gpd.read_parquet(cache_edges)
        else:
            pbf = next(RAW.glob("*.osm.pbf"), None)
            if pbf is None: raise FileNotFoundError("required OSM PBF missing")
            nodes, edges, stats["walking_graph"] = build_walking_graph(pbf, cfg, cache_edges)
    if "walking_graph" not in stats:
        stats["walking_graph"] = {"nodes": len(nodes), "directed_edges": len(edges), "cached": True}
    region_bbox = cfg["city_bbox"]
    all_stops=[]; all_routes=[]; all_trips=[]; all_stop_times=[]; all_connections=[]; quarantine=[]
    for feed, mode in zip(feeds, ("BUS", "METRO")):
        parsed = parse_feed(feed, mode, service_date, region_bbox)
        stops, routes, trips, st, conns, bad = parsed
        all_stops.append(stops); all_routes.append(routes.assign(agency=mode, mode=mode)); all_trips.append(trips)
        all_stop_times.append(st); all_connections.append(conns)
        if not bad.empty:
            bad["agency"] = mode
            quarantine.append(bad)
    stops = pd.concat(all_stops, ignore_index=True)
    routes = pd.concat(all_routes, ignore_index=True)
    trips = pd.concat(all_trips, ignore_index=True)
    stop_times = pd.concat(all_stop_times, ignore_index=True)
    # IDs must be unique across agencies; feed-local integer IDs can overlap.
    stops["feed_stop_id"] = stops.stop_id.astype(str)
    stops["stop_idx"] = np.arange(len(stops), dtype=np.int64)
    stop_key = {(r.agency, str(r.stop_id)): int(r.stop_idx) for r in stops.itertuples()}
    trips["trip_idx"] = np.arange(len(trips), dtype=np.int64)
    trip_key = {(r.agency, str(r.trip_id)): int(r.trip_idx) for r in trips.itertuples()}
    route_key = {(r.agency, str(r.route_id)): i for i, r in enumerate(routes.itertuples())}
    routes["route_idx"] = np.arange(len(routes), dtype=np.int64)
    route_key = {(r.agency, str(r.route_id)): int(r.route_idx) for r in routes.itertuples()}
    trips["route_idx"] = [route_key[(r.agency, str(r.route_id))] for r in trips.itertuples()]
    stop_times["trip_idx"] = [trip_key[(r.agency, str(r.trip_id))] for r in stop_times.itertuples()]
    stop_times["stop_idx"] = [stop_key[(r.agency, str(r.stop_id))] for r in stop_times.itertuples()]
    stop_times = stop_times.sort_values(["trip_idx", "stop_sequence"], kind="stable")
    connections = stop_times.copy()
    connections["from_stop"] = connections.groupby("trip_idx").stop_idx.shift()
    connections["dep_time"] = connections.groupby("trip_idx").departure_sec.shift()
    connections["to_stop"] = connections.stop_idx
    connections["arr_time"] = connections.arrival_sec
    connections = connections[connections.from_stop.notna()].copy()
    connections["from_stop"] = connections.from_stop.astype(np.int64)
    transformer = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
    xy = np.column_stack(transformer.transform(nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()))
    tree = cKDTree(xy)
    sx, sy = transformer.transform(stops.lon.to_numpy(), stops.lat.to_numpy())
    distances, indices = tree.query(np.column_stack([sx, sy]), k=1)
    stops["node_idx"] = nodes.iloc[indices].node_idx.to_numpy()
    stops["snap_dist_m"] = distances
    stops["poorly_snapped"] = distances > float(cfg["stop_snap_max_m"])
    edges, safety_cov = add_safety(nodes, edges, cfg)
    edges.to_parquet(cache_edges, index=False)
    nodes.to_parquet(cache_nodes, index=False)
    for name, frame in [("stops",stops),("routes",routes),("trips",trips),("stop_times",stop_times),("connections",connections)]:
        frame = frame.copy()
        # GTFS often mixes blank strings and numeric values in optional columns.
        for column in frame.select_dtypes(include="object").columns:
            frame[column] = frame[column].astype("string")
        frame.to_parquet(OUT / f"{name}.parquet", index=False)
    if quarantine: pd.concat(quarantine, ignore_index=True).to_csv(ROOT/"derived_data"/"quarantine"/"invalid_stops.csv", index=False)
    cov = {"service_date": service_date, "walking": stats["walking_graph"],
        "transit": {"stops": int(len(stops)), "bus_stops": int(sum(stops["mode"]=="BUS")), "metro_stops": int(sum(stops["mode"]=="METRO")),
            "routes": int(len(routes)), "trips": int(len(trips)), "connections": int(len(connections)),
            "poorly_snapped_stops": int(stops.poorly_snapped.sum()), "poorly_snapped_share": float(stops.poorly_snapped.mean())},
        "safety": safety_cov,
        "limitations": ["Accident data is aggregate and not assigned to edges", "Streetlight CSV is ward-level; OSM lamps are used only when mapped",
            "Bus/metro fare and emission units require task-defined placeholder treatment where feeds are ambiguous", "Metro stop times may be approximate"]}
    stats.update(cov)
    (OUT / "coverage.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(cov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
