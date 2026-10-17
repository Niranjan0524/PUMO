"""Algorithm A4: Greedy Insertion Heuristic (GIH) for dynamic paratransit routing."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from .models import Stop, StopType, PassengerRequest, Vehicle, AssignmentDecision
from .dijkstra import DijkstraRouter
from .constraints import RouteConstraintChecker
from .fare import ParatransitFareCalculator


class GreedyInsertionAssigner:
    """Algorithm A4: Evaluates all feasible pickup/drop-off insertion positions across all vehicles.

    Inserts the pickup stop at index i and dropoff stop at index j (i <= j)
    into the vehicle's planned route, verifying capacity, precedence, time windows,
    and preservation of commitments to existing passengers.
    """

    def __init__(
        self,
        router: DijkstraRouter,
        fare_calc: Optional[ParatransitFareCalculator] = None,
        weight_vehicle_time: float = 1.0,
        weight_wait_time: float = 0.8,
        weight_detour: float = 1.0,
        weight_delay_to_others: float = 1.5,
    ):
        self.router = router
        self.fare_calc = fare_calc or ParatransitFareCalculator()
        self.checker = RouteConstraintChecker(router)
        self.w_veh = weight_vehicle_time
        self.w_wait = weight_wait_time
        self.w_detour = weight_detour
        self.w_delay = weight_delay_to_others

    def find_all_feasible_insertions(
        self,
        request: PassengerRequest,
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
    ) -> List[AssignmentDecision]:
        """Finds all feasible (vehicle, pickup_idx, dropoff_idx) combinations for a request."""
        feasible_insertions: List[AssignmentDecision] = []

        for veh in vehicles:
            if request.num_passengers > veh.capacity:
                continue

            # Compute current route baseline time
            _, _, base_veh_time, base_veh_dist, _ = self.checker.simulate_schedule(
                start_node=veh.current_node,
                start_time=veh.current_time,
                initial_load=veh.occupied_seats,
                capacity=veh.capacity,
                stops=veh.route,
            )

            p_stop = Stop(
                stop_id=f"stop_{request.request_id}_p",
                node_id=request.origin_id,
                request_id=request.request_id,
                stop_type=StopType.PICKUP,
                passengers_delta=request.num_passengers,
            )
            d_stop = Stop(
                stop_id=f"stop_{request.request_id}_d",
                node_id=request.destination_id,
                request_id=request.request_id,
                stop_type=StopType.DROPOFF,
                passengers_delta=-request.num_passengers,
            )

            m = len(veh.route)
            # Try all positions: 0 <= i <= j <= m
            for i in range(m + 1):
                # Quick bounding check: if dispatch from route[i-1] to origin alone exceeds max wait, skip
                for j in range(i, m + 1):
                    # Construct candidate stop sequence:
                    # insert p_stop at i, then d_stop at j+1
                    candidate_stops = list(veh.route)
                    candidate_stops.insert(i, p_stop)
                    candidate_stops.insert(j + 1, d_stop)

                    is_feasible, sched_stops, metrics, reason = self.checker.evaluate_candidate_route(
                        vehicle=veh,
                        candidate_stops=candidate_stops,
                        new_request=request,
                        active_requests=active_requests,
                    )

                    if not is_feasible:
                        continue

                    # Calculate marginal cost
                    delta_veh_time = max(0.0, metrics["total_route_time"] - base_veh_time)
                    delta_veh_dist = max(0.0, metrics["total_route_dist"] - base_veh_dist)
                    wait_time = metrics["wait_time"]
                    ride_time = metrics["in_vehicle_time"]
                    journey_time = wait_time + ride_time
                    delay_existing = metrics["delay_to_existing"]
                    detour_sec = max(0.0, ride_time - request.direct_travel_time)

                    # Compute concurrency and fare
                    approx_concurrency = max(1.0, 1.0 + (len(veh.route) * 0.4))
                    fare = self.fare_calc.compute_fare(
                        direct_distance_km=request.direct_distance_km,
                        direct_time_sec=request.direct_travel_time,
                        actual_time_sec=ride_time,
                        avg_concurrent_passengers=approx_concurrency,
                        num_passengers=request.num_passengers,
                    )

                    cost = (
                        (self.w_veh * delta_veh_time)
                        + (self.w_wait * wait_time)
                        + (self.w_detour * detour_sec)
                        + (self.w_delay * delay_existing)
                    )

                    decision = AssignmentDecision(
                        vehicle_id=veh.vehicle_id,
                        pickup_index=i,
                        dropoff_index=j + 1,
                        is_feasible=True,
                        delta_vehicle_time=delta_veh_time,
                        delta_vehicle_distance=delta_veh_dist,
                        passenger_wait_time=wait_time,
                        passenger_journey_time=journey_time,
                        passenger_fare=fare,
                        delay_to_existing=delay_existing,
                        score=cost,
                        objective_vector=(journey_time, fare),
                    )
                    feasible_insertions.append(decision)

        return feasible_insertions

    def assign(
        self,
        request: PassengerRequest,
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
    ) -> AssignmentDecision:
        """Finds the single best greedy insertion minimizing marginal combined cost."""
        feasible_insertions = self.find_all_feasible_insertions(
            request=request,
            vehicles=vehicles,
            active_requests=active_requests,
            current_time=current_time,
        )

        if not feasible_insertions:
            return AssignmentDecision(
                is_feasible=False,
                rejection_reason="No feasible insertion satisfying capacity and time-window constraints",
            )

        # Pick insertion with lowest score
        feasible_insertions.sort(key=lambda d: d.score)
        return feasible_insertions[0]
