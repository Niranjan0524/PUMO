"""Benchmark evaluation comparing all paratransit routing algorithms across demand tiers."""
from __future__ import annotations

import json
import os
import sys
import time
import pandas as pd

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pumo_paratransit.models import PassengerRequest, Vehicle
from pumo_paratransit.network import RoadGraph
from pumo_paratransit.dijkstra import DijkstraRouter
from pumo_paratransit.fare import ParatransitFareCalculator
from pumo_paratransit.simulator import ParatransitSimulator
from pumo_paratransit.baselines import NearestAvailableAssigner, NearestFeasibleAssigner
from pumo_paratransit.insertion import GreedyInsertionAssigner
from pumo_paratransit.regret import RegretInsertionAssigner
from pumo_paratransit.pareto import ParetoMultiObjectiveAssigner


def load_scenario(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    requests = [
        PassengerRequest(
            request_id=r["request_id"],
            origin_id=r["origin_id"],
            destination_id=r["destination_id"],
            request_time=r["request_time"],
            num_passengers=r["num_passengers"],
            max_wait_time=r["max_wait_time"],
            max_detour_ratio=r["max_detour_ratio"],
            direct_travel_time=r.get("direct_travel_time", 0.0),
            direct_distance_km=r.get("direct_distance_km", 0.0),
        )
        for r in data["requests"]
    ]

    vehicles = [
        Vehicle(
            vehicle_id=v["vehicle_id"],
            capacity=v["capacity"],
            current_node=v["current_node"],
            current_time=v["current_time"],
            occupied_seats=v["occupied_seats"],
        )
        for v in data["vehicles"]
    ]

    return data["scenario_name"], requests, vehicles


def run_benchmark():
    data_dir = os.path.join(parent_dir, "data")
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    net_path = os.path.join(data_dir, "synthetic_network.json")
    graph = RoadGraph.load_json(net_path)
    router = DijkstraRouter(graph)
    print("Precomputing all-pairs shortest paths...")
    router.precompute_all_pairs()
    fare_calc = ParatransitFareCalculator()
    simulator = ParatransitSimulator(router)

    algorithms = [
        ("A2_NearestAvailable", NearestAvailableAssigner(router, fare_calc)),
        ("A3_NearestFeasible", NearestFeasibleAssigner(router, fare_calc)),
        ("A4_GreedyInsertion", GreedyInsertionAssigner(router, fare_calc)),
        ("A5_RegretInsertion", RegretInsertionAssigner(router, fare_calc, k=2)),
        ("A6_Pareto_TimeFirst", ParetoMultiObjectiveAssigner(router, fare_calc, strategy="time_first")),
        ("A6_Pareto_FareFirst", ParetoMultiObjectiveAssigner(router, fare_calc, strategy="fare_first")),
        ("A6_Pareto_Balanced", ParetoMultiObjectiveAssigner(router, fare_calc, strategy="balanced")),
    ]

    scenarios = [
        os.path.join(data_dir, "scenario_low.json"),
        os.path.join(data_dir, "scenario_medium.json"),
        os.path.join(data_dir, "scenario_high.json"),
    ]

    benchmark_rows = []

    print("\n" + "=" * 90)
    print(f"{'Scenario':<16} | {'Algorithm':<22} | {'SrvRate':<8} | {'Wait(min)':<9} | {'Ride(min)':<9} | {'Fare(Rs)':<8} | {'Detour':<7} | {'FleetKm':<8} | {'ms/req':<7}")
    print("=" * 90)

    for scn_file in scenarios:
        scn_name, reqs, vehs = load_scenario(scn_file)

        for algo_name, assigner in algorithms:
            metrics, sim_reqs, sim_vehs = simulator.run_simulation(
                algorithm_name=algo_name,
                assigner=assigner,
                requests=reqs,
                vehicles=vehs,
                precompute_direct=True,
            )

            row = metrics.to_dict()
            row["scenario"] = scn_name
            benchmark_rows.append(row)

            print(
                f"{scn_name:<16} | "
                f"{algo_name:<22} | "
                f"{row['service_rate_pct']:>6.1f}% | "
                f"{row['avg_wait_time_min']:>9.2f} | "
                f"{row['avg_in_vehicle_time_sec'] / 60.0:>9.2f} | "
                f"Rs{row['avg_fare']:>6.1f} | "
                f"{row['avg_detour_ratio']:>7.2f} | "
                f"{row['total_vehicle_dist_km']:>8.1f} | "
                f"{row['avg_runtime_ms']:>7.2f}"
            )

    print("=" * 90 + "\n")

    # Export to CSV and JSON
    df = pd.DataFrame(benchmark_rows)
    # Reorder columns
    cols = [
        "scenario", "algorithm", "total_requests", "served_requests", "rejected_requests",
        "service_rate_pct", "avg_wait_time_min", "avg_in_vehicle_time_sec", "avg_journey_time_min",
        "avg_detour_ratio", "avg_fare", "total_fare_revenue", "total_vehicle_dist_km",
        "total_runtime_ms", "avg_runtime_ms"
    ]
    df = df[cols]

    csv_path = os.path.join(results_dir, "benchmark_summary.csv")
    json_path = os.path.join(results_dir, "benchmark_results.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_rows, f, indent=2)

    print(f"Benchmark summary saved to:\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    run_benchmark()
