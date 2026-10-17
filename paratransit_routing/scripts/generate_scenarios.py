"""Generates reproducible paratransit test scenarios (Low, Medium, High, Exact Benchmark)."""
from __future__ import annotations

import json
import random
import os
import sys

# Ensure src is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pumo_paratransit.network import create_bengaluru_synthetic_network, create_small_test_graph
from pumo_paratransit.dijkstra import DijkstraRouter


def generate_scenario(
    name: str,
    graph,
    router,
    num_requests: int,
    num_vehicles: int,
    duration_sec: float = 3600.0,
    vehicle_capacity: int = 4,
    seed: int = 42,
) -> dict:
    random.seed(seed)
    node_ids = list(graph.nodes.keys())

    # Generate initial vehicle placements distributed across nodes
    vehicles = []
    for i in range(num_vehicles):
        vid = f"V{i+1:02d}"
        init_node = node_ids[i % len(node_ids)]
        vehicles.append({
            "vehicle_id": vid,
            "capacity": vehicle_capacity,
            "current_node": init_node,
            "current_time": 0.0,
            "occupied_seats": 0,
        })

    # Generate dynamic requests with Poisson / uniform arrival
    requests = []
    current_t = 30.0
    for i in range(num_requests):
        rid = f"R{i+1:03d}"
        # Advance arrival time
        inter_arrival = random.expovariate(num_requests / duration_sec)
        current_t += inter_arrival
        if current_t > duration_sec:
            current_t = duration_sec - random.uniform(5.0, 60.0)

        # Select origin and destination (ensure distinct)
        origin = random.choice(node_ids)
        dest_candidates = [n for n in node_ids if n != origin]
        destination = random.choice(dest_candidates)

        direct_tt = router.get_travel_time(origin, destination)
        direct_dist = router.get_distance(origin, destination)

        # Party size: 75% 1 pax, 20% 2 pax, 5% 3 pax
        party_draw = random.random()
        if party_draw < 0.75:
            num_pax = 1
        elif party_draw < 0.95:
            num_pax = 2
        else:
            num_pax = 3

        # Realistic wait and detour tolerances
        max_wait = round(random.uniform(600.0, 900.0), 1)  # 10 to 15 mins
        max_detour = round(random.uniform(1.5, 1.8), 2)     # 1.5x to 1.8x

        requests.append({
            "request_id": rid,
            "origin_id": origin,
            "destination_id": destination,
            "request_time": round(current_t, 1),
            "num_passengers": num_pax,
            "max_wait_time": max_wait,
            "max_detour_ratio": max_detour,
            "direct_travel_time": round(direct_tt, 2),
            "direct_distance_km": round(direct_dist, 2),
        })

    # Sort requests by arrival time
    requests.sort(key=lambda r: r["request_time"])

    return {
        "scenario_name": name,
        "seed": seed,
        "duration_sec": duration_sec,
        "num_requests": len(requests),
        "num_vehicles": len(vehicles),
        "vehicles": vehicles,
        "requests": requests,
    }


def main():
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Save Bengaluru synthetic network
    graph = create_bengaluru_synthetic_network()
    net_path = os.path.join(data_dir, "synthetic_network.json")
    graph.save_json(net_path)
    print(f"Saved road network to {net_path} ({len(graph.nodes)} nodes, {sum(len(edges) for edges in graph.adjacency.values())} edges)")

    router = DijkstraRouter(graph)
    router.precompute_all_pairs()

    # 2. Generate Low Demand (20 requests, 4 vehicles)
    low_data = generate_scenario("Scenario_Low", graph, router, num_requests=20, num_vehicles=4, seed=101)
    with open(os.path.join(data_dir, "scenario_low.json"), "w", encoding="utf-8") as f:
        json.dump(low_data, f, indent=2)
    print(f"Saved scenario_low.json: 20 requests, 4 vehicles")

    # 3. Generate Medium Demand (60 requests, 8 vehicles)
    med_data = generate_scenario("Scenario_Medium", graph, router, num_requests=60, num_vehicles=8, seed=202)
    with open(os.path.join(data_dir, "scenario_medium.json"), "w", encoding="utf-8") as f:
        json.dump(med_data, f, indent=2)
    print(f"Saved scenario_medium.json: 60 requests, 8 vehicles")

    # 4. Generate High Demand (150 requests, 12 vehicles)
    high_data = generate_scenario("Scenario_High", graph, router, num_requests=150, num_vehicles=12, seed=303)
    with open(os.path.join(data_dir, "scenario_high.json"), "w", encoding="utf-8") as f:
        json.dump(high_data, f, indent=2)
    print(f"Saved scenario_high.json: 150 requests, 12 vehicles")

    # 5. Generate Exact Benchmark Scenario (small instance: 6 requests, 2 vehicles on small graph)
    small_graph = create_small_test_graph()
    small_router = DijkstraRouter(small_graph)
    small_router.precompute_all_pairs()
    exact_data = generate_scenario(
        "Scenario_Exact_Validation",
        small_graph,
        small_router,
        num_requests=6,
        num_vehicles=2,
        duration_sec=1200.0,
        vehicle_capacity=3,
        seed=404,
    )
    with open(os.path.join(data_dir, "scenario_exact.json"), "w", encoding="utf-8") as f:
        json.dump(exact_data, f, indent=2)
    print(f"Saved scenario_exact.json: 6 requests, 2 vehicles")


if __name__ == "__main__":
    main()
