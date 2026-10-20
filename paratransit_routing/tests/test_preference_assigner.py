"""Unit tests for PreferenceWeightedAssigner (grounded in Papers 2 & 3)."""
from __future__ import annotations
import os
import sys
import unittest

test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pumo_paratransit.models import PassengerRequest, Vehicle, Stop, StopType
from pumo_paratransit.network import create_small_test_graph
from pumo_paratransit.dijkstra import DijkstraRouter
from pumo_paratransit.passenger_preference import PassengerSegment, PassengerUtility
from pumo_paratransit.preference_assigner import PreferenceWeightedAssigner


class TestPreferenceWeightedAssigner(unittest.TestCase):

    def setUp(self):
        self.graph = create_small_test_graph()
        self.router = DijkstraRouter(self.graph)
        self.router.precompute_all_pairs()

        self.req = PassengerRequest(
            request_id="R_PREF",
            origin_id="A",
            destination_id="C",
            request_time=100.0,
            num_passengers=1,
            max_wait_time=900.0,
            max_detour_ratio=2.0,
            direct_travel_time=self.router.get_travel_time("A", "C"),
            direct_distance_km=self.router.get_distance("A", "C"),
        )

    def test_assigner_finds_feasible_assignment(self):
        v = Vehicle("V01", capacity=3, current_node="A", current_time=100.0)
        assigner = PreferenceWeightedAssigner(self.router, segment=PassengerSegment.OCCASIONAL_USER)
        decision = assigner.assign(self.req, [v], {}, current_time=100.0)

        self.assertTrue(decision.accepted)
        self.assertGreater(decision.score, 0.0)  # Generalised cost > 0
        self.assertEqual(decision.vehicle_id, "V01")

    def test_crowding_aversion_prefers_less_crowded_vehicle(self):
        """Safety-conscious / crowding-sensitive passenger prefers empty vehicle over crowded vehicle (Paper 3)."""
        # V1: 2 seats occupied out of 3 (occupancy = 66.7% before + 1 = 100% full -> heavy crowding penalty)
        v1 = Vehicle("V_CROWDED", capacity=3, current_node="A", current_time=100.0, occupied_seats=2)
        v1.route = [Stop("s_drop_v1", "C", "R_EXIST", StopType.DROPOFF, -2, planned_arrival_time=940.0)]

        # V2: 0 seats occupied (empty vehicle at A)
        v2 = Vehicle("V_EMPTY", capacity=3, current_node="A", current_time=100.0, occupied_seats=0)

        assigner = PreferenceWeightedAssigner(self.router, segment=PassengerSegment.SAFETY_CONSCIOUS)
        decision = assigner.assign(self.req, [v1, v2], {"R_EXIST": PassengerRequest("R_EXIST", "A", "C", 0.0, 2)}, current_time=100.0)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.vehicle_id, "V_EMPTY", "Should pick uncrowded vehicle to avoid crowding penalty")

    def test_time_sensitive_vs_price_sensitive_segment_behavior(self):
        """Validates that different segments evaluate the trade-off differently."""
        v = Vehicle("V01", capacity=3, current_node="A", current_time=100.0)

        assigner_time = PreferenceWeightedAssigner(self.router, segment=PassengerSegment.TIME_SENSITIVE)
        assigner_reg = PreferenceWeightedAssigner(self.router, segment=PassengerSegment.REGULAR_COMMUTER)

        dec_time = assigner_time.assign(self.req, [v], {}, current_time=100.0)
        dec_reg = assigner_reg.assign(self.req, [v], {}, current_time=100.0)

        self.assertTrue(dec_time.accepted)
        self.assertTrue(dec_reg.accepted)
        # Because time-sensitive passenger values time much higher, their generalised cost score should differ
        self.assertNotEqual(dec_time.score, dec_reg.score)


if __name__ == "__main__":
    unittest.main()
