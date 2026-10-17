"""Compares heuristics A2, A3, A4, A5, and A6 against Exact Solver A7 to compute optimality gaps."""
from __future__ import annotations

import json
import os
import sys
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
from pumo_paratransit.exact_solver import ExactDARPSolver
from pumo_paratransit.baselines import NearestAvailableAssigner, NearestFeasibleAssigner
from pumo_paratransit.insertion import GreedyInsertionAssigner
from pumo_paratransit.regret import RegretInsertionAssigner
from pumo_paratransit.pareto import ParetoMultiObjectiveAssigner
from pumo_paratransit.simulator import ParatransitSimulator


def run_exact_gap_evaluation():
    data_dir = os.path.join(parent_dir, "data")
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    exact_file = os.path.join(data_dir, "scenario_exact.json")
    with open(exact_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct small validation graph
    from pumo_paratransit.network import create_small_test_graph
    graph = create_small_test_graph()
    router = DijkstraRouter(graph)
    router.precompute_all_pairs()
    fare_calc = ParatransitFareCalculator()

    requests = [
        PassengerRequest(
            request_id=r["request_id"],
            origin_id=r["origin_id"],
            destination_id=r["destination_id"],
            request_time=r["request_time"],
            num_passengers=r["num_passengers"],
            max_wait_time=r["max_wait_time"],
            max_detour_ratio=r["max_detour_ratio"],
            direct_travel_time=router.get_travel_time(r["origin_id"], r["destination_id"]),
            direct_distance_km=router.get_distance(r["origin_id"], r["destination_id"]),
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

    print("\n" + "=" * 80)
    print("RUNNING EXACT OPTIMIZATION SOLVER (Algorithm A7) BENCHMARK")
    print("=" * 80)

    # 1. Exact Global Optimum (A7)
    exact_solver = ExactDARPSolver(router, fare_calc, weight_journey_time=1.0, weight_vehicle_dist=0.5)
    exact_solution = exact_solver.solve(requests, vehicles, current_time=0.0)
    exact_cost = exact_solution["best_cost"]

    print(f"Algorithm A7 (Exact Solver):")
    print(f"  Proven Global Optimal Cost: {exact_cost:.2f}")
    print(f"  Served Requests:            {len(exact_solution['served_requests'])}/{len(requests)}")
    print(f"  Total Passenger Journey:    {exact_solution['total_journey_time']:.1f} s")
    print(f"  Total Vehicle Distance:     {exact_solution['total_distance_km']:.1f} km")
    print("-" * 80)

    # 2. Heuristics evaluated on the exact scenario
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

    comparison_rows = []
    # Add A7 exact record
    comparison_rows.append({
        "algorithm": "A7_ExactSolver (Global Optimum)",
        "served_requests": len(exact_solution["served_requests"]),
        "total_requests": len(requests),
        "service_rate_pct": (len(exact_solution["served_requests"]) / len(requests)) * 100.0,
        "total_journey_time_s": exact_solution["total_journey_time"],
        "total_distance_km": exact_solution["total_distance_km"],
        "combined_cost": exact_cost,
        "optimality_gap_pct": 0.0,
    })

    for algo_name, assigner in algorithms:
        metrics, sim_reqs, sim_vehs = simulator.run_simulation(
            algorithm_name=algo_name,
            assigner=assigner,
            requests=requests,
            vehicles=vehicles,
        )

        served = [r for r in sim_reqs if r.pickup_time is not None and r.dropoff_time is not None]
        unserved_count = len(sim_reqs) - len(served)
        journey_sum = sum(r.total_journey_time for r in served)
        dist_sum = sum(v.total_distance_km for v in sim_vehs)
        cost = (1.0 * journey_sum) + (0.5 * dist_sum) + (unserved_count * exact_solver.penalty_unserved)

        gap_pct = ((cost - exact_cost) / exact_cost) * 100.0 if exact_cost > 0 else 0.0

        comparison_rows.append({
            "algorithm": algo_name,
            "served_requests": len(served),
            "total_requests": len(requests),
            "service_rate_pct": metrics.service_rate_pct,
            "total_journey_time_s": round(journey_sum, 1),
            "total_distance_km": round(dist_sum, 1),
            "combined_cost": round(cost, 2),
            "optimality_gap_pct": round(gap_pct, 2),
        })

    df = pd.DataFrame(comparison_rows)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    # Save to CSV and JSON
    csv_path = os.path.join(results_dir, "exact_optimality_gap.csv")
    json_path = os.path.join(results_dir, "exact_optimality_gap.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2)

    print(f"Exact optimality gap results saved to:\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    run_exact_gap_evaluation()
