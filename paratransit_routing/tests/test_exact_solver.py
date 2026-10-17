"""Unit tests for Exact DARP Solver (Algorithm A7) on small instances."""
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

from pumo_paratransit.models import PassengerRequest, Vehicle
from pumo_paratransit.network import create_small_test_graph
from pumo_paratransit.dijkstra import DijkstraRouter
from pumo_paratransit.exact_solver import ExactDARPSolver
from pumo_paratransit.insertion import GreedyInsertionAssigner


class TestExactSolver(unittest.TestCase):

    def setUp(self):
        self.graph = create_small_test_graph()
        self.router = DijkstraRouter(self.graph)
        self.exact_solver = ExactDARPSolver(self.router)
        self.greedy = GreedyInsertionAssigner(self.router)

    def test_exact_solver_global_optimum(self):
        # 3 requests, 1 vehicle
        v = Vehicle("V01", capacity=3, current_node="A", current_time=0.0)

        requests = [
            PassengerRequest(
                request_id="R1",
                origin_id="A",
                destination_id="B",
                request_time=0.0,
                num_passengers=1,
                max_wait_time=600.0,
                max_detour_ratio=1.5,
                direct_travel_time=self.router.get_travel_time("A", "B"),
                direct_distance_km=self.router.get_distance("A", "B"),
            ),
            PassengerRequest(
                request_id="R2",
                origin_id="B",
                destination_id="C",
                request_time=100.0,
                num_passengers=1,
                max_wait_time=600.0,
                max_detour_ratio=1.5,
                direct_travel_time=self.router.get_travel_time("B", "C"),
                direct_distance_km=self.router.get_distance("B", "C"),
            ),
        ]

        solution = self.exact_solver.solve(requests, [v])

        self.assertTrue(solution["optimality_proven"])
        self.assertEqual(len(solution["served_requests"]), 2)
        self.assertLess(solution["best_cost"], float("inf"))
        self.assertGreater(len(solution["vehicle_routes"]["V01"]), 0)


if __name__ == "__main__":
    unittest.main()
