"""Unit tests for baseline assigners: Nearest Available (A2) and Nearest Feasible (A3)."""
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
from pumo_paratransit.baselines import NearestAvailableAssigner, NearestFeasibleAssigner


class TestBaselines(unittest.TestCase):

    def setUp(self):
        self.graph = create_small_test_graph()
        self.router = DijkstraRouter(self.graph)
        self.nav = NearestAvailableAssigner(self.router)
        self.nfv = NearestFeasibleAssigner(self.router)

        self.req = PassengerRequest(
            request_id="R01",
            origin_id="B",
            destination_id="D",
            request_time=100.0,
            num_passengers=1,
            max_wait_time=800.0,
            max_detour_ratio=1.6,
            direct_travel_time=self.router.get_travel_time("B", "D"),
            direct_distance_km=self.router.get_distance("B", "D"),
        )

    def test_nearest_available_idle_only(self):
        # Veh 1 is busy (carrying 1 pax), Veh 2 is idle at A (near B)
        v1 = Vehicle("V01", capacity=3, current_node="B", occupied_seats=1)
        v2 = Vehicle("V02", capacity=3, current_node="A", occupied_seats=0)

        dec = self.nav.assign(self.req, [v1, v2], {}, current_time=100.0)
        self.assertTrue(dec.accepted)
        # Must pick V02 because V01 is busy (NAV does not share)
        self.assertEqual(dec.vehicle_id, "V02")

    def test_nearest_available_rejection_when_all_busy(self):
        v1 = Vehicle("V01", capacity=3, current_node="B", occupied_seats=1)
        v2 = Vehicle("V02", capacity=3, current_node="A", occupied_seats=2)

        dec = self.nav.assign(self.req, [v1, v2], {}, current_time=100.0)
        self.assertFalse(dec.accepted)
        self.assertIn("No idle vehicle", dec.rejection_reason)

    def test_nearest_feasible_can_share(self):
        # V01 is at B, carrying 1 pax (capacity 3)
        v1 = Vehicle("V01", capacity=3, current_node="B", occupied_seats=1)
        dec = self.nfv.assign(self.req, [v1], {}, current_time=100.0)
        self.assertTrue(dec.accepted)
        self.assertEqual(dec.vehicle_id, "V01")


if __name__ == "__main__":
    unittest.main()
