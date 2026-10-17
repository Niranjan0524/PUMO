"""End-to-end dynamic simulation tests across algorithms."""
from __future__ import annotations

import unittest
import os
import sys

# Add src to sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pumo_paratransit.models import PassengerRequest, Vehicle, RequestStatus
from pumo_paratransit.network import create_small_test_graph
from pumo_paratransit.dijkstra import DijkstraRouter
from pumo_paratransit.simulator import ParatransitSimulator
from pumo_paratransit.baselines import NearestAvailableAssigner, NearestFeasibleAssigner
from pumo_paratransit.insertion import GreedyInsertionAssigner
from pumo_paratransit.pareto import ParetoMultiObjectiveAssigner


class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.graph = create_small_test_graph()
        self.router = DijkstraRouter(self.graph)
        self.simulator = ParatransitSimulator(self.router)

        self.vehicles = [
            Vehicle("V01", capacity=3, current_node="A", current_time=0.0),
            Vehicle("V02", capacity=3, current_node="C", current_time=0.0),
        ]

        self.requests = [
            PassengerRequest(
                request_id="R01",
                origin_id="A",
                destination_id="B",
                request_time=10.0,
                num_passengers=1,
                max_wait_time=600.0,
                max_detour_ratio=1.6,
            ),
            PassengerRequest(
                request_id="R02",
                origin_id="B",
                destination_id="C",
                request_time=50.0,
                num_passengers=1,
                max_wait_time=600.0,
                max_detour_ratio=1.6,
            ),
            PassengerRequest(
                request_id="R03",
                origin_id="C",
                destination_id="D",
                request_time=120.0,
                num_passengers=1,
                max_wait_time=600.0,
                max_detour_ratio=1.6,
            ),
        ]

    def test_simulation_run_greedy(self):
        assigner = GreedyInsertionAssigner(self.router)
        metrics, processed_reqs, processed_vehs = self.simulator.run_simulation(
            algorithm_name="Greedy_Insertion",
            assigner=assigner,
            requests=self.requests,
            vehicles=self.vehicles,
        )

        self.assertGreaterEqual(metrics.served_requests, 2)
        self.assertGreater(metrics.service_rate_pct, 50.0)
        self.assertGreater(metrics.total_fare_revenue, 0.0)

        # Check completed requests have recorded times
        completed = [r for r in processed_reqs if r.status == RequestStatus.COMPLETED]
        for r in completed:
            self.assertIsNotNone(r.pickup_time)
            self.assertIsNotNone(r.dropoff_time)
            self.assertGreater(r.dropoff_time, r.pickup_time)

    def test_simulation_run_pareto_balanced(self):
        assigner = ParetoMultiObjectiveAssigner(self.router, strategy="balanced")
        metrics, processed_reqs, processed_vehs = self.simulator.run_simulation(
            algorithm_name="Pareto_Balanced",
            assigner=assigner,
            requests=self.requests,
            vehicles=self.vehicles,
        )

        self.assertGreaterEqual(metrics.served_requests, 2)
        self.assertGreater(metrics.avg_fare, 0.0)


if __name__ == "__main__":
    unittest.main()
