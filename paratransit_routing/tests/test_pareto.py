"""Unit tests for Pareto Multi-Objective Optimization (Algorithm A6)."""
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

from pumo_paratransit.models import PassengerRequest, Vehicle, Stop, StopType
from pumo_paratransit.network import create_small_test_graph
from pumo_paratransit.dijkstra import DijkstraRouter
from pumo_paratransit.pareto import ParetoMultiObjectiveAssigner, dominates


class TestParetoMultiObjective(unittest.TestCase):

    def setUp(self):
        self.graph = create_small_test_graph()
        self.router = DijkstraRouter(self.graph)

    def test_dominance_relation(self):
        # a is strictly better than b on one objective, equal on other
        self.assertTrue(dominates((10.0, 50.0), (12.0, 50.0)))
        # a is better on both
        self.assertTrue(dominates((8.0, 40.0), (10.0, 50.0)))
        # Trade-off: a is faster, but more expensive -> neither dominates
        self.assertFalse(dominates((8.0, 60.0), (10.0, 50.0)))
        self.assertFalse(dominates((10.0, 50.0), (8.0, 60.0)))

    def test_pareto_front_generation(self):
        # Create two vehicles with different trade-offs:
        # V1: Idle at A (fast, direct, solo fare)
        v1 = Vehicle("V01", capacity=3, current_node="A", current_time=0.0)
        # V2: Busy vehicle at B with an active passenger (sharing discount, higher detour)
        v2 = Vehicle("V02", capacity=3, current_node="B", current_time=0.0, occupied_seats=1)
        v2.route = [Stop("s0", "D", "R_PREV", StopType.DROPOFF, -1, 720.0, 750.0)]

        req = PassengerRequest(
            request_id="R_TEST",
            origin_id="B",
            destination_id="C",
            request_time=0.0,
            num_passengers=1,
            max_wait_time=900.0,
            max_detour_ratio=2.0,
            direct_travel_time=self.router.get_travel_time("B", "C"),
            direct_distance_km=self.router.get_distance("B", "C"),
        )

        assigner = ParetoMultiObjectiveAssigner(self.router)
        front = assigner.compute_pareto_front(req, [v1, v2], {}, current_time=0.0)

        self.assertGreaterEqual(len(front), 1)

        # Time-first must pick lowest journey time
        best_time = assigner.select_from_pareto_front(front, strategy="time_first")
        self.assertIsNotNone(best_time)

        # Fare-first must pick lowest fare
        best_fare = assigner.select_from_pareto_front(front, strategy="fare_first")
        self.assertIsNotNone(best_fare)
        self.assertLessEqual(best_fare.passenger_fare, best_time.passenger_fare)

        # Balanced pick
        best_balanced = assigner.select_from_pareto_front(front, strategy="balanced")
        self.assertIsNotNone(best_balanced)


if __name__ == "__main__":
    unittest.main()
