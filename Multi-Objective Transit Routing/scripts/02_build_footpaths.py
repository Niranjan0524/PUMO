#!/usr/bin/env python3
"""Precompute bounded Pareto stop-to-stop pedestrian transfer options."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pumo.walking import WalkingGraph


def main():
    cfg = yaml.safe_load((ROOT / "config" / "thresholds.yaml").read_text(encoding="utf-8"))
    p = ROOT / "derived_data" / "processed"
    edges = gpd.read_parquet(p / "walking_edges.parquet")
    nodes = gpd.read_parquet(p / "walking_nodes.parquet")
    stops = pd.read_parquet(p / "stops.parquet")
    graph = WalkingGraph(len(nodes), edges.from_node, edges.to_node, edges.edge_id,
        edges.walk_time_s, edges.length_m, edges.length_m * edges.safety_risk.fillna(0))
    metric = Transformer.from_crs("EPSG:4326", cfg["metric_crs"], always_xy=True)
    node_xy = np.column_stack(metric.transform(nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()))
    node_pos = {int(n): i for i, n in enumerate(nodes.node_idx)}
    stop_nodes = stops.node_idx.astype(int).to_numpy()
    stop_xy = node_xy[np.fromiter((node_pos[n] for n in stop_nodes), dtype=np.int64, count=len(stop_nodes))]
    stop_tree = cKDTree(stop_xy)
    per_node = {}
    for i, node in enumerate(stop_nodes): per_node.setdefault(int(node), []).append(i)
    rows = []
    unique_sources = sorted(per_node)
    for count, source in enumerate(unique_sources, 1):
        source_stop_indexes = per_node[source]
        nearby = stop_tree.query_ball_point(stop_xy[source_stop_indexes[0]], r=float(cfg["transfer_radius_m"]))
        target_nodes = set(int(stop_nodes[i]) for i in nearby if int(stop_nodes[i]) != source)
        if not target_nodes: continue
        options = graph.bounded_options(source, target_nodes, max_distance_m=float(cfg["transfer_radius_m"]),
            max_time_s=int(cfg["max_footpath_time_s"]), limit=int(cfg["walking_option_limit"]), labels_per_node=12)
        by_target = {}
        for option in options: by_target.setdefault(option.target, []).append(option)
        for target, target_stops in ((target, per_node[target]) for target in by_target):
            if target == source: continue
            for left in source_stop_indexes:
                for right in target_stops:
                    for option_idx, opt in enumerate(by_target[target][:int(cfg["walking_option_limit"])]):
                        rows.append({"from_stop": int(stops.iloc[left].stop_idx), "to_stop": int(stops.iloc[right].stop_idx),
                            "walk_time_s": opt.walk_time_s, "distance_m": opt.distance_m,
                            "risk_exposure": opt.risk_exposure, "edge_ids": list(opt.edge_ids),
                            "option_idx": option_idx, "source_type": "DERIVED"})
        if count % 1000 == 0: print(f"searched {count}/{len(unique_sources)} distinct stop nodes; options={len(rows)}")
    result = pd.DataFrame(rows)
    result.to_parquet(p / "footpaths.parquet", index=False)
    coverage_path = ROOT / "derived_data" / "processed" / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["footpaths"] = {"options": int(len(result)), "distinct_stop_nodes": len(unique_sources),
        "stop_pairs": int(result[["from_stop", "to_stop"]].drop_duplicates().shape[0]) if len(result) else 0,
        "K": int(cfg["walking_option_limit"]), "radius_m": float(cfg["transfer_radius_m"]),
        "max_time_s": int(cfg["max_footpath_time_s"]), "per_node_label_cap": 12,
        "approximation": "bounded non-dominated labels per node; output options are Pareto-filtered"}
    coverage_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(json.dumps(coverage["footpaths"], indent=2))


if __name__ == "__main__":
    main()
