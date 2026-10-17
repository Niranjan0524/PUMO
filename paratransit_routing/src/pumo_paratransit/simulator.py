"""Dynamic event-driven simulation engine for paratransit routing."""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple, Any
from .models import Stop, StopType, PassengerRequest, RequestStatus, Vehicle, AssignmentDecision
from .dijkstra import DijkstraRouter
from .constraints import RouteConstraintChecker
from .metrics import SimulationMetrics, compute_simulation_metrics


class ParatransitSimulator:
    """Dynamic discrete-event simulator for on-demand shared-auto routing."""

    def __init__(self, router: DijkstraRouter):
        self.router = router
        self.checker = RouteConstraintChecker(router)

    def advance_vehicle_to_time(
        self,
        vehicle: Vehicle,
        target_time: float,
        active_requests: Dict[str, PassengerRequest],
    ) -> None:
        """Simulates vehicle progress along its planned route up to target_time."""
        while vehicle.route:
            next_stop = vehicle.route[0]

            # If the next stop finishes on or before target_time, complete it
            if next_stop.planned_departure_time <= target_time:
                # Vehicle moves to this stop
                travel_dist = self.router.get_distance(vehicle.current_node, next_stop.node_id)
                travel_time = self.router.get_travel_time(vehicle.current_node, next_stop.node_id)
                vehicle.total_distance_km += travel_dist
                vehicle.total_travel_time_sec += travel_time

                vehicle.current_node = next_stop.node_id
                vehicle.current_time = next_stop.planned_departure_time

                rid = next_stop.request_id
                if rid in active_requests:
                    req = active_requests[rid]
                    if next_stop.stop_type == StopType.PICKUP:
                        req.status = RequestStatus.IN_TRANSIT
                        req.pickup_time = next_stop.planned_arrival_time
                        vehicle.occupied_seats += next_stop.passengers_delta
                    elif next_stop.stop_type == StopType.DROPOFF:
                        req.status = RequestStatus.COMPLETED
                        req.dropoff_time = next_stop.planned_arrival_time
                        vehicle.occupied_seats += next_stop.passengers_delta  # negative delta
                        vehicle.served_requests_count += 1
                        del active_requests[rid]

                vehicle.route.pop(0)
            else:
                # Stop is not yet finished at target_time
                break

        if not vehicle.route:
            vehicle.current_time = max(vehicle.current_time, target_time)

    def run_simulation(
        self,
        algorithm_name: str,
        assigner: Any,
        requests: List[PassengerRequest],
        vehicles: List[Vehicle],
        precompute_direct: bool = True,
    ) -> Tuple[SimulationMetrics, List[PassengerRequest], List[Vehicle]]:
        """Executes a dynamic simulation run for a given assignment algorithm.

        Returns:
            Tuple of:
              - SimulationMetrics
              - Processed requests
              - Final vehicle states
        """
        # Deep copy to maintain scenario isolation
        sim_requests = [copy.deepcopy(r) for r in requests]
        sim_vehicles = [copy.deepcopy(v) for v in vehicles]
        vehicle_map = {v.vehicle_id: v for v in sim_vehicles}

        # Precompute direct shortest paths
        if precompute_direct:
            for r in sim_requests:
                if r.direct_travel_time <= 0:
                    r.direct_travel_time = self.router.get_travel_time(r.origin_id, r.destination_id)
                if r.direct_distance_km <= 0:
                    r.direct_distance_km = self.router.get_distance(r.origin_id, r.destination_id)

        # Sort requests chronologically by arrival time
        sim_requests.sort(key=lambda r: r.request_time)

        active_requests: Dict[str, PassengerRequest] = {}
        start_cpu = time.perf_counter()

        for req in sim_requests:
            curr_time = req.request_time

            # 1. Advance all vehicles up to current request arrival time
            for veh in sim_vehicles:
                self.advance_vehicle_to_time(veh, curr_time, active_requests)

            # 2. Query assignment algorithm
            decision: AssignmentDecision = assigner.assign(
                request=req,
                vehicles=sim_vehicles,
                active_requests=active_requests,
                current_time=curr_time,
            )

            # 3. Apply assignment or record rejection
            if decision.accepted:
                veh = vehicle_map[decision.vehicle_id]
                req.status = RequestStatus.ASSIGNED
                req.assigned_vehicle_id = veh.vehicle_id
                req.fare = decision.passenger_fare

                # Build stop objects
                p_stop = Stop(
                    stop_id=f"stop_{req.request_id}_p",
                    node_id=req.origin_id,
                    request_id=req.request_id,
                    stop_type=StopType.PICKUP,
                    passengers_delta=req.num_passengers,
                )
                d_stop = Stop(
                    stop_id=f"stop_{req.request_id}_d",
                    node_id=req.destination_id,
                    request_id=req.request_id,
                    stop_type=StopType.DROPOFF,
                    passengers_delta=-req.num_passengers,
                )

                # Insert stops into vehicle route
                p_idx = decision.pickup_index
                d_idx = decision.dropoff_index
                veh.route.insert(p_idx, p_stop)
                veh.route.insert(d_idx, d_stop)

                # Re-simulate vehicle schedule to establish updated arrival/departure times
                _, sched_stops, _, _, _ = self.checker.simulate_schedule(
                    start_node=veh.current_node,
                    start_time=veh.current_time,
                    initial_load=veh.occupied_seats,
                    capacity=veh.capacity,
                    stops=veh.route,
                )
                veh.route = sched_stops
                active_requests[req.request_id] = req

            else:
                req.status = RequestStatus.REJECTED
                req.rejection_reason = decision.rejection_reason or "Assignment infeasible"

        # 4. End of arrivals: advance all vehicles to finish remaining commitments
        max_horizon = 100000.0
        for veh in sim_vehicles:
            self.advance_vehicle_to_time(veh, max_horizon, active_requests)

        end_cpu = time.perf_counter()
        runtime_ms = (end_cpu - start_cpu) * 1000.0

        metrics = compute_simulation_metrics(
            algorithm_name=algorithm_name,
            requests=sim_requests,
            vehicles=sim_vehicles,
            runtime_ms=runtime_ms,
        )

        return metrics, sim_requests, sim_vehicles
