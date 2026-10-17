"""Unit tests for D-DARP route constraints (precedence, capacity, time windows, and preservation)."""
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

from pumo_paratransit.models import Stop, StopType, PassengerRequest, Vehicle, RequestStatus
from pumo_paratransit.network import create_small_test_graph
from pumo_paratransit.dijkstra import DijkstraRouter
from pumo_paratransit.constraints import RouteConstraintChecker


class TestConstraints(unittest.TestCase):

    def setUp(self):
        self.graph = create_small_test_graph()
        self.router = DijkstraRouter(self.graph)
        self.checker = RouteConstraintChecker(self.router)

        # Vehicle at node A, capacity 3
        self.veh = Vehicle(
            vehicle_id="V01",
            capacity=3,
            current_node="A",
            current_time=0.0,
            occupied_seats=0,
        )

        # Request R01 from A to C
        self.req = PassengerRequest(
            request_id="R01",
            origin_id="A",
            destination_id="C",
            request_time=0.0,
            num_passengers=2,
            max_wait_time=500.0,
            max_detour_ratio=1.5,
            direct_travel_time=self.router.get_travel_time("A", "C"),  # 840 s
            direct_distance_km=self.router.get_distance("A", "C"),
        )

    def test_precedence_violation(self):
        """Dropoff before pickup must be rejected."""
        p_stop = Stop("s_p", "A", "R01", StopType.PICKUP, 2)
        d_stop = Stop("s_d", "C", "R01", StopType.DROPOFF, -2)

        # Inverted order: dropoff before pickup
        candidate = [d_stop, p_stop]
        is_feas, _, _, reason = self.checker.evaluate_candidate_route(
            vehicle=self.veh,
            candidate_stops=candidate,
            new_request=self.req,
            active_requests={},
        )
        self.assertFalse(is_feas)
        self.assertIn("Precedence violation", reason)

    def test_capacity_violation(self):
        """Vehicle capacity of 3 cannot accept 4 passengers."""
        oversized_req = PassengerRequest(
            request_id="R_BIG",
            origin_id="A",
            destination_id="C",
            request_time=0.0,
            num_passengers=4,  # > vehicle capacity 3
            max_wait_time=500.0,
            max_detour_ratio=1.5,
            direct_travel_time=840.0,
        )
        p_stop = Stop("s_p", "A", "R_BIG", StopType.PICKUP, 4)
        d_stop = Stop("s_d", "C", "R_BIG", StopType.DROPOFF, -4)

        is_feas, _, _, reason = self.checker.evaluate_candidate_route(
            vehicle=self.veh,
            candidate_stops=[p_stop, d_stop],
            new_request=oversized_req,
            active_requests={},
        )
        self.assertFalse(is_feas)
        self.assertIn("Capacity exceeded", reason)

    def test_wait_time_violation(self):
        """Pickup scheduled after max wait time must be rejected."""
        # Vehicle starts at E (far away: E -> D -> C -> B -> A is > 1000s)
        far_veh = Vehicle("V_FAR", capacity=3, current_node="E", current_time=0.0)
        # Tight wait time of only 200s
        tight_req = PassengerRequest(
            request_id="R_TIGHT",
            origin_id="A",
            destination_id="B",
            request_time=0.0,
            num_passengers=1,
            max_wait_time=200.0,
            max_detour_ratio=1.5,
            direct_travel_time=360.0,
        )
        p_stop = Stop("s_p", "A", "R_TIGHT", StopType.PICKUP, 1)
        d_stop = Stop("s_d", "B", "R_TIGHT", StopType.DROPOFF, -1)

        is_feas, _, _, reason = self.checker.evaluate_candidate_route(
            vehicle=far_veh,
            candidate_stops=[p_stop, d_stop],
            new_request=tight_req,
            active_requests={},
        )
        self.assertFalse(is_feas)
        self.assertIn("Wait time", reason)

    def test_existing_passenger_preservation(self):
        """Cannot insert a new request if it pushes an existing passenger beyond their promised deadline."""
        # Existing passenger R00 on board, heading to C with tight max detour
        req_existing = PassengerRequest(
            request_id="R00",
            origin_id="A",
            destination_id="C",
            request_time=0.0,
            num_passengers=1,
            max_wait_time=300.0,
            max_detour_ratio=1.1,  # tight: 840 * 1.1 = 924 s max ride
            direct_travel_time=840.0,
            pickup_time=0.0,
            dropoff_time=840.0,
        )
        # Vehicle is at B, carrying R00, scheduled to drop off at C
        veh = Vehicle("V_MID", capacity=3, current_node="B", current_time=360.0, occupied_seats=1)
        veh.route = [Stop("s_d0", "C", "R00", StopType.DROPOFF, -1, planned_arrival_time=840.0)]

        # New request wants a big detour to E before dropping off R00
        req_new = PassengerRequest(
            request_id="R_DETOUR",
            origin_id="E",
            destination_id="D",
            request_time=360.0,
            num_passengers=1,
            max_wait_time=3000.0,
            max_detour_ratio=4.0,
            direct_travel_time=600.0,
        )

        p_new = Stop("p_new", "E", "R_DETOUR", StopType.PICKUP, 1)
        d_new = Stop("d_new", "D", "R_DETOUR", StopType.DROPOFF, -1)

        # Candidate inserting E then D BEFORE C (violating R00's tight deadline)
        candidate = [p_new, d_new, veh.route[0]]
        is_feas, _, _, reason = self.checker.evaluate_candidate_route(
            vehicle=veh,
            candidate_stops=candidate,
            new_request=req_new,
            active_requests={"R00": req_existing},
        )
        self.assertFalse(is_feas)
        self.assertIn("Violates existing passenger", reason)


if __name__ == "__main__":
    unittest.main()
