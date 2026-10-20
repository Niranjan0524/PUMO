"""Runs empirical evaluation comparing preference-weighted assignment across passenger segments (Papers 2 & 3)."""
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
from pumo_paratransit.simulator import ParatransitSimulator
from pumo_paratransit.passenger_preference import PassengerSegment, PassengerUtility
from pumo_paratransit.preference_assigner import PreferenceWeightedAssigner
from pumo_paratransit.insertion import GreedyInsertionAssigner
from pumo_paratransit.baselines import NearestAvailableAssigner, NearestFeasibleAssigner
from pumo_paratransit.gtfs_model import create_bengaluru_shared_auto_network


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


def run_preference_benchmark():
    data_dir = os.path.join(parent_dir, "data")
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save GTFS shared-auto network if not already present
    gtfs_net = create_bengaluru_shared_auto_network()
    gtfs_path = os.path.join(data_dir, "bengaluru_shared_auto_gtfs.json")
    gtfs_net.save_json(gtfs_path)

    net_path = os.path.join(data_dir, "synthetic_network.json")
    graph = RoadGraph.load_json(net_path)
    router = DijkstraRouter(graph)
    router.precompute_all_pairs()
    fare_calc = ParatransitFareCalculator()
    pref_util = PassengerUtility()
    simulator = ParatransitSimulator(router)

    scenario_file = os.path.join(data_dir, "scenario_medium.json")
    scn_name, reqs, vehs = load_scenario(scenario_file)

    assigners = [
        ("A2: Solo Ride-Hail (NAV)", NearestAvailableAssigner(router, fare_calc), None),
        ("A3: Sequential Append (NFV)", NearestFeasibleAssigner(router, fare_calc), None),
        ("A4: Standard Greedy (GIH)", GreedyInsertionAssigner(router, fare_calc), None),
        ("Pref: Regular Commuter", PreferenceWeightedAssigner(router, fare_calc, segment=PassengerSegment.REGULAR_COMMUTER), PassengerSegment.REGULAR_COMMUTER),
        ("Pref: Time-Sensitive", PreferenceWeightedAssigner(router, fare_calc, segment=PassengerSegment.TIME_SENSITIVE), PassengerSegment.TIME_SENSITIVE),
        ("Pref: Safety-Conscious", PreferenceWeightedAssigner(router, fare_calc, segment=PassengerSegment.SAFETY_CONSCIOUS), PassengerSegment.SAFETY_CONSCIOUS),
        ("Pref: Occasional User", PreferenceWeightedAssigner(router, fare_calc, segment=PassengerSegment.OCCASIONAL_USER), PassengerSegment.OCCASIONAL_USER),
    ]

    rows = []

    print("\n" + "=" * 105)
    print("EVALUATION OF PASSENGER PREFERENCE MODELS (Das et al. 2023 & Das & Mandal 2021)")
    print("=" * 105)
    print(f"{'Strategy / Segment':<28} | {'SrvRate':<8} | {'Wait(min)':<9} | {'Ride(min)':<9} | {'Fare(Rs)':<8} | {'Detour':<7} | {'Avg GC (Rs)':<11} | {'Avg Utility':<11}")
    print("-" * 105)

    for label, assigner, seg in assigners:
        eval_seg = seg or PassengerSegment.OCCASIONAL_USER
        metrics, sim_reqs, sim_vehs = simulator.run_simulation(
            algorithm_name=label,
            assigner=assigner,
            requests=reqs,
            vehicles=vehs,
            precompute_direct=True,
        )

        served = [r for r in sim_reqs if r.pickup_time is not None and r.dropoff_time is not None]

        # Calculate average generalised cost and systematic utility for served passengers
        gc_list = []
        utility_list = []
        for r in served:
            wait_m = (r.waiting_time or 0.0) / 60.0
            ride_m = (r.in_vehicle_time or 0.0) / 60.0
            # Default access time: 2.0 min
            access_m = 2.0
            occ_ratio = 0.5  # average during trip
            gc = pref_util.compute_generalised_cost_inr(
                in_vehicle_time_min=ride_m,
                waiting_time_min=wait_m,
                access_time_min=access_m,
                fare_inr=r.fare,
                occupancy_ratio=occ_ratio,
                has_realtime_info=True,
                segment=eval_seg,
            )
            u = pref_util.compute_utility(
                in_vehicle_time_min=ride_m,
                waiting_time_min=wait_m,
                access_time_min=access_m,
                fare_inr=r.fare,
                occupancy_ratio=occ_ratio,
                has_realtime_info=True,
                segment=eval_seg,
            )
            gc_list.append(gc)
            utility_list.append(u)

        avg_gc = (sum(gc_list) / len(gc_list)) if gc_list else 0.0
        avg_u = (sum(utility_list) / len(utility_list)) if utility_list else 0.0

        row = {
            "strategy": label,
            "segment": eval_seg.value,
            "service_rate_pct": metrics.service_rate_pct,
            "avg_wait_min": metrics.avg_wait_time_min,
            "avg_ride_min": metrics.avg_in_vehicle_time_sec / 60.0,
            "avg_fare_inr": metrics.avg_fare,
            "avg_detour": metrics.avg_detour_ratio,
            "fleet_km": metrics.total_vehicle_distance_km,
            "avg_generalised_cost_inr": round(avg_gc, 2),
            "avg_systematic_utility": round(avg_u, 3),
        }
        rows.append(row)

        print(
            f"{label:<28} | "
            f"{metrics.service_rate_pct:>6.1f}% | "
            f"{metrics.avg_wait_time_min:>9.2f} | "
            f"{metrics.avg_in_vehicle_time_sec / 60.0:>9.2f} | "
            f"Rs{metrics.avg_fare:>6.1f} | "
            f"{metrics.avg_detour_ratio:>7.2f} | "
            f"Rs{avg_gc:>9.2f} | "
            f"{avg_u:>11.3f}"
        )

    print("=" * 105 + "\n")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "paper_preference_summary.csv")
    json_path = os.path.join(results_dir, "paper_preference_summary.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Results saved to:\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    run_preference_benchmark()
