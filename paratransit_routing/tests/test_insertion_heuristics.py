"""Unit tests for Greedy Insertion (A4) and Regret-k Insertion (A5)."""
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

from pumo_paratransit.models import Stop, StopType, PassengerRequest, Vehicle
from pumo_paratransit.network import create_small_test_graph
from pumo_paratransit.dijkstra import DijkstraRouter
from pumo_paratransit.insertion import GreedyInsertionAssigner
from pumo_paratransit.regret import RegretInsertionAssigner


class TestInsertionHeuristics(unittest.TestCase):

    def setUp(self):
        self.graph = create_small_test_graph()
        self.router = DijkstraRouter(self.graph)
        self.greedy = GreedyInsertionAssigner(self.router)
        self.regret = RegretInsertionAssigner(self.router)

        self.r1 = PassengerRequest(
            request_id="R01",
            origin_id="A",
            destination_id="D",
            request_time=0.0,
            num_passengers=1,
            max_wait_time=900.0,
            max_detour_ratio=1.8,
            direct_travel_time=self.router.get_travel_time("A", "D"),
            direct_distance_km=self.router.get_distance("A", "D"),
        )
        self.r2 = PassengerRequest(
            request_id="R02",
            origin_id="B",
            destination_id="C",
            request_time=50.0,
            num_passengers=1,
            max_wait_time=600.0,
            max_detour_ratio=1.5,
            direct_travel_time=self.router.get_travel_time("B", "C"),
            direct_distance_km=self.router.get_distance("B", "C"),
        )

    def test_greedy_insertion_finds_best_position(self):
        v = Vehicle("V01", capacity=3, current_node="A", current_time=0.0)
        dec1 = self.greedy.assign(self.r1, [v], {}, current_time=0.0)
        self.assertTrue(dec1.accepted)
        self.assertEqual(dec1.pickup_index, 0)
        self.assertEqual(dec1.dropoff_index, 1)

    def test_greedy_interleaved_insertion(self):
        # Vehicle already committed to take R01 from A to D (path A -> B -> C -> D)
        # New request R02 is from B to C (which lies along the exact path!)
        # Greedy should insert R02 between A and D cleanly
        v = Vehicle("V01", capacity=3, current_node="A", current_time=0.0)
        p1 = Stop("s_p1", "A", "R01", StopType.PICKUP, 1, 0.0, 30.0)
        d1 = Stop("s_d1", "D", "R01", StopType.DROPOFF, -1, 1200.0, 1230.0)
        v.route = [p1, d1]

        dec2 = self.greedy.assign(self.r2, [v], {"R01": self.r1}, current_time=50.0)
        self.assertTrue(dec2.accepted)
        # Should insert R02 between p1 and d1
        self.assertGreaterEqual(dec2.pickup_index, 1)
        self.assertLessEqual(dec2.dropoff_index, 2)

    def test_regret_insertion_prioritization(self):
        v1 = Vehicle("V01", capacity=3, current_node="A", current_time=0.0)
        v2 = Vehicle("V02", capacity=3, current_node="D", current_time=0.0)

        # R1 is at A (near V01, far from V02 -> high regret)
        # R2 is at B (accessible to both -> lower regret)
        batch = [self.r2, self.r1]
        assignments = self.regret.assign_batch(batch, [v1, v2], {}, current_time=50.0)
        self.assertEqual(len(assignments), 2)
        self.assertTrue(assignments["R01"].accepted)
        self.assertTrue(assignments["R02"].accepted)


if __name__ == "__main__":
    unittest.main()
