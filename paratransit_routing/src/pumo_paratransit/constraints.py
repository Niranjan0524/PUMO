"""Constraint checking and schedule simulation for D-DARP vehicle routes."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from .models import Stop, StopType, PassengerRequest, Vehicle, RequestStatus
from .dijkstra import DijkstraRouter


class RouteConstraintChecker:
    """Enforces capacity, precedence, time-window, and existing-passenger preservation constraints."""

    def __init__(self, router: DijkstraRouter):
        self.router = router

    def simulate_schedule(
        self,
        start_node: str,
        start_time: float,
        initial_load: int,
        capacity: int,
        stops: List[Stop]
    ) -> Tuple[bool, List[Stop], float, float, Optional[str]]:
        """Simulates vehicle progression through a stop sequence.

        Returns:
            Tuple of:
              - is_valid (bool): True if capacity and physical rules hold
              - scheduled_stops (List[Stop]): Stops updated with arrival/departure times
              - total_travel_time (float): Sum of pure vehicle travel time in seconds
              - total_distance_km (float): Sum of vehicle distance traveled in km
              - failure_reason (Optional[str])
        """
        curr_node = start_node
        curr_time = start_time
        curr_load = initial_load

        if curr_load > capacity:
            return False, [], 0.0, 0.0, f"Initial load {curr_load} exceeds vehicle capacity {capacity}"

        scheduled_stops: List[Stop] = []
        total_travel_time = 0.0
        total_distance = 0.0

        for stop in stops:
            travel_time = self.router.get_travel_time(curr_node, stop.node_id)
            distance = self.router.get_distance(curr_node, stop.node_id)

            if travel_time == float("inf"):
                return False, [], 0.0, 0.0, f"Unreachable node {stop.node_id} from {curr_node}"

            arrival_time = curr_time + travel_time
            departure_time = arrival_time + stop.service_duration

            new_load = curr_load + stop.passengers_delta
            if new_load > capacity:
                return False, [], 0.0, 0.0, f"Capacity exceeded at stop {stop.node_id}: {new_load} > {capacity}"
            if new_load < 0:
                return False, [], 0.0, 0.0, f"Negative load at stop {stop.node_id}: {new_load}"

            sched_stop = Stop(
                stop_id=stop.stop_id,
                node_id=stop.node_id,
                request_id=stop.request_id,
                stop_type=stop.stop_type,
                passengers_delta=stop.passengers_delta,
                planned_arrival_time=arrival_time,
                planned_departure_time=departure_time,
                service_duration=stop.service_duration,
            )
            scheduled_stops.append(sched_stop)

            curr_node = stop.node_id
            curr_time = departure_time
            curr_load = new_load
            total_travel_time += travel_time
            total_distance += distance

        return True, scheduled_stops, total_travel_time, total_distance, None

    def evaluate_candidate_route(
        self,
        vehicle: Vehicle,
        candidate_stops: List[Stop],
        new_request: PassengerRequest,
        active_requests: Dict[str, PassengerRequest],
    ) -> Tuple[bool, List[Stop], Dict[str, float], Optional[str]]:
        """Validates all D-DARP constraints on a candidate stop sequence.

        Checks:
          1. Vehicle capacity along the entire sequence.
          2. Precedence: Pickup before dropoff for all requests.
          3. New passenger time windows (maximum wait time and maximum detour ratio).
          4. Existing passenger preservation (no already promised deadlines are broken).

        Returns:
            Tuple of:
              - is_feasible (bool)
              - scheduled_stops (List[Stop])
              - metrics (Dict with wait time, in-vehicle time, detour, delay to others)
              - failure_reason (Optional[str])
        """
        # Precedence check across all requests in candidate stops
        pickup_pos: Dict[str, int] = {}
        dropoff_pos: Dict[str, int] = {}
        for idx, s in enumerate(candidate_stops):
            if s.stop_type == StopType.PICKUP:
                pickup_pos[s.request_id] = idx
            elif s.stop_type == StopType.DROPOFF:
                dropoff_pos[s.request_id] = idx

        for req_id, p_idx in pickup_pos.items():
            if req_id in dropoff_pos and p_idx > dropoff_pos[req_id]:
                return False, [], {}, f"Precedence violation: pickup after dropoff for request {req_id}"

        # Determine vehicle dispatch start time:
        # If vehicle has no existing stops, it starts at max(vehicle.current_time, new_request.request_time)
        start_time = max(vehicle.current_time, new_request.request_time) if len(vehicle.route) == 0 else vehicle.current_time

        # Simulate timeline & capacity
        is_valid, sched_stops, total_tt, total_dist, reason = self.simulate_schedule(
            start_node=vehicle.current_node,
            start_time=start_time,
            initial_load=vehicle.occupied_seats,
            capacity=vehicle.capacity,
            stops=candidate_stops,
        )

        if not is_valid:
            return False, [], {}, reason

        # Map stop schedule by request
        sched_pickup_arr: Dict[str, float] = {}
        sched_pickup_dep: Dict[str, float] = {}
        sched_dropoff_arr: Dict[str, float] = {}

        for s in sched_stops:
            if s.stop_type == StopType.PICKUP:
                sched_pickup_arr[s.request_id] = s.planned_arrival_time
                sched_pickup_dep[s.request_id] = s.planned_departure_time
            elif s.stop_type == StopType.DROPOFF:
                sched_dropoff_arr[s.request_id] = s.planned_arrival_time

        # Check new request feasibility
        new_rid = new_request.request_id
        if new_rid not in sched_pickup_arr or new_rid not in sched_dropoff_arr:
            return False, [], {}, "New request stops missing from candidate route"

        new_pickup_arr = sched_pickup_arr[new_rid]
        new_dropoff_arr = sched_dropoff_arr[new_rid]
        new_wait = max(0.0, new_pickup_arr - new_request.request_time)

        # Check wait time constraint
        if new_wait > new_request.max_wait_time:
            return False, [], {}, f"Wait time {new_wait:.1f}s exceeds max {new_request.max_wait_time:.1f}s"

        # Check detour constraint
        new_ride_time = new_dropoff_arr - sched_pickup_dep[new_rid]
        max_allowed_ride = new_request.direct_travel_time * new_request.max_detour_ratio
        if new_ride_time > max_allowed_ride:
            return False, [], {}, f"Ride time {new_ride_time:.1f}s exceeds max allowed detour {max_allowed_ride:.1f}s"

        # Check preservation of existing passengers
        delay_to_existing = 0.0
        for rid, req in active_requests.items():
            if rid == new_rid:
                continue

            # If existing request has a planned pickup in this vehicle's stops
            if rid in sched_pickup_arr:
                p_arr = sched_pickup_arr[rid]
                ex_wait = p_arr - req.request_time
                if ex_wait > req.max_wait_time:
                    return False, [], {}, f"Violates existing passenger {rid} max wait deadline"

            # Check existing dropoff deadline
            if rid in sched_dropoff_arr:
                d_arr = sched_dropoff_arr[rid]
                # Determine baseline pickup reference time
                if req.pickup_time is not None:
                    pickup_ref = req.pickup_time
                elif rid in sched_pickup_dep:
                    pickup_ref = sched_pickup_dep[rid]
                else:
                    pickup_ref = req.request_time

                ex_ride_time = d_arr - pickup_ref
                ex_max_ride = req.direct_travel_time * req.max_detour_ratio
                if ex_ride_time > ex_max_ride:
                    return False, [], {}, f"Violates existing passenger {rid} max detour deadline"

                # Calculate delay added to existing passenger
                if req.dropoff_time is not None and d_arr > req.dropoff_time:
                    delay_to_existing += (d_arr - req.dropoff_time)

        metrics = {
            "wait_time": new_wait,
            "in_vehicle_time": new_ride_time,
            "total_journey_time": new_wait + new_ride_time,
            "detour_ratio": (new_ride_time / new_request.direct_travel_time) if new_request.direct_travel_time > 0 else 1.0,
            "delay_to_existing": delay_to_existing,
            "total_route_time": total_tt,
            "total_route_dist": total_dist,
        }

        return True, sched_stops, metrics, None
