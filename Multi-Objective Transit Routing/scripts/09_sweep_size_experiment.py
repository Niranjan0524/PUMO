#!/usr/bin/env python3
"""Measure W sweep-size sensitivity on seeded short/medium/long City queries."""
from __future__ import annotations

import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pumo.algos.weight_sweep import WeightSweepRouter
from pumo.model import Query
from pumo.network import PreparedNetwork
from pumo.objectives import OBJECTIVES


def main():
    cfg = yaml.safe_load((ROOT / "config" / "benchmark.yaml").read_text(encoding="utf-8"))
    qpath = ROOT / "derived_data" / "benchmark" / "queries_city.parquet"
    queries = pd.read_parquet(qpath)
    selected = []
    for band in ("short", "medium", "long"):
        group = queries[queries.distance_band.eq(band)].sort_values("query_id")
        if len(group):
            selected.append(group.iloc[0])
    if len(selected) != 3:
        raise SystemExit("Sweep experiment requires seeded short, medium, and long City queries")

    net_started = time.perf_counter()
    net = PreparedNetwork.from_directory(ROOT / "derived_data" / "processed", scenario="city",
                                         boarding_buffer_s=30)
    preprocessing_s = time.perf_counter() - net_started
    profiles_frame = pd.read_csv(ROOT / "config" / "profiles.csv")
    profiles = [{key: float(row[col]) for key, col in zip(OBJECTIVES, profiles_frame.columns[1:])}
                for _, row in profiles_frame.iterrows()]
    rows = []
    timeout_s = 5.0
    label_cap = 10_000
    sizes = (10, 25, 50)
    for row in selected:
        query = Query((float(row.origin_lat), float(row.origin_lon)),
                      (float(row.destination_lat), float(row.destination_lon)),
                      int(row.departure_time), OBJECTIVES, timeout_s=timeout_s,
                      max_transfers=int(cfg["max_transfers"]))
        for size in sizes:
            router = WeightSweepRouter(net, scales=cfg["scales"], sweep_size=size,
                                       seed=int(cfg["seed"]), profiles=profiles,
                                       label_cap=label_cap,
                                       max_speed_cap_mps=float(cfg.get("max_speed_cap_mps", 40.0)))
            started = time.perf_counter()
            result = router.route(query)
            elapsed = time.perf_counter() - started
            rows.append({
                "query_id": row.query_id,
                "distance_band": row.distance_band,
                "seed": int(row.seed),
                "objective_set": "".join(OBJECTIVES),
                "sweep_size_requested": size,
                "sweep_size_actual": result.stats.get("sweep_size"),
                "weight_vectors_run": result.stats.get("weight_vectors_run"),
                "runtime_s": elapsed,
                "router_runtime_s": result.stats.get("runtime_s"),
                "journeys_returned": len(result.journeys),
                "labels_generated": result.stats.get("labels_generated", 0),
                "labels_expanded": result.stats.get("labels_expanded", 0),
                "timed_out": result.stats.get("timed_out", False),
                "label_cap_hit": result.stats.get("label_cap_hit", False),
                "exact": result.stats.get("exact", False),
                "supported_front_only": result.stats.get("supported_front_only", True),
            })

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "sweep_size_city.csv", index=False)
    try:
        revision = __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        revision = None
    packages = {}
    for package in ("numpy", "pandas", "pyarrow", "geopandas", "shapely", "pyproj", "scipy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    metadata = {
        "protocol_version": "1.5",
        "experiment": "W sweep-size sensitivity",
        "scenario": "city",
        "query_count": len(selected),
        "query_ids": [str(row.query_id) for row in selected],
        "objective_set": "".join(OBJECTIVES),
        "sweep_sizes_requested": list(sizes),
        "timeout_s": timeout_s,
        "label_cap": label_cap,
        "repetitions": 1,
        "seed": int(cfg["seed"]),
        "network_preprocessing_s": preprocessing_s,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "git_revision": revision,
        "package_versions": packages,
    }
    (out / "sweep_size_city_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
