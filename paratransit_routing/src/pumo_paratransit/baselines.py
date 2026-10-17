"""Algorithms A2 and A3: Baseline paratransit assignment heuristics."""
from __future__ import annotations

from typing import Dict, List, Optional
from .models import Stop, StopType, PassengerRequest, Vehicle, AssignmentDecision
from .dijkstra import DijkstraRouter
from .constraints import RouteConstraintChecker
from .fare import ParatransitFareCalculator


class NearestAvailableAssigner:
    """Algorithm A2: Assigns the nearest available (idle/unshared) vehicle.

    Represents standard solo on-demand ride hailing: vehicles only serve one
    passenger party at a time. No ride sharing.
    """

    def __init__(self, router: DijkstraRouter, fare_calc: Optional[ParatransitFareCalculator] = None):
        self.router = router
        self.fare_calc = fare_calc or ParatransitFareCalculator()
        self.checker = RouteConstraintChecker(router)

    def assign(
        self,
        request: PassengerRequest,
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
    ) -> AssignmentDecision:
        best_decision = AssignmentDecision(is_feasible=False, rejection_reason="No idle vehicle within wait limit")
        min_dispatch_time = float("inf")

        for veh in vehicles:
            if not veh.is_idle:
                continue

            # Check capacity
            if request.num_passengers > veh.capacity:
                continue

            dispatch_time = self.router.get_travel_time(veh.current_node, request.origin_id)
            if dispatch_time == float("inf"):
                continue

            pickup_arr = current_time + dispatch_time
            wait_time = pickup_arr - request.request_time

            if wait_time > request.max_wait_time:
                continue

            if dispatch_time < min_dispatch_time:
                min_dispatch_time = dispatch_time

                pickup_stop = Stop(
                    stop_id=f"stop_{request.request_id}_p",
                    node_id=request.origin_id,
                    request_id=request.request_id,
                    stop_type=StopType.PICKUP,
                    passengers_delta=request.num_passengers,
                )
                dropoff_stop = Stop(
                    stop_id=f"stop_{request.request_id}_d",
                    node_id=request.destination_id,
                    request_id=request.request_id,
                    stop_type=StopType.DROPOFF,
                    passengers_delta=-request.num_passengers,
                )

                candidate_stops = [pickup_stop, dropoff_stop]
                is_feasible, sched_stops, metrics, reason = self.checker.evaluate_candidate_route(
                    vehicle=veh,
                    candidate_stops=candidate_stops,
                    new_request=request,
                    active_requests=active_requests,
                )

                if is_feasible:
                    fare = self.fare_calc.compute_solo_fare(request.direct_distance_km)
                    best_decision = AssignmentDecision(
                        vehicle_id=veh.vehicle_id,
                        pickup_index=0,
                        dropoff_index=1,
                        is_feasible=True,
                        delta_vehicle_time=dispatch_time + request.direct_travel_time,
                        delta_vehicle_distance=self.router.get_distance(veh.current_node, request.origin_id) + request.direct_distance_km,
                        passenger_wait_time=wait_time,
                        passenger_journey_time=wait_time + request.direct_travel_time,
                        passenger_fare=fare,
                        delay_to_existing=0.0,
                        score=dispatch_time,
                        rejection_reason=None,
                        objective_vector=(wait_time + request.direct_travel_time, fare),
                    )

        return best_decision


class NearestFeasibleAssigner:
    """Algorithm A3: Assigns the nearest feasible vehicle (sequential append).

    Permits vehicle sharing by assigning to vehicles with spare capacity,
    appending the new trip to the end of the vehicle's planned route without
    complex re-sequencing.
    """

    def __init__(self, router: DijkstraRouter, fare_calc: Optional[ParatransitFareCalculator] = None):
        self.router = router
        self.fare_calc = fare_calc or ParatransitFareCalculator()
        self.checker = RouteConstraintChecker(router)

    def assign(
        self,
        request: PassengerRequest,
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
    ) -> AssignmentDecision:
        best_decision = AssignmentDecision(is_feasible=False, rejection_reason="No feasible vehicle for sequential append")
        best_score = float("inf")

        for veh in vehicles:
            if request.num_passengers > veh.capacity:
                continue

            # Candidate stops: append pickup then dropoff at the end of current route
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

            p_idx = len(veh.route)
            d_idx = len(veh.route) + 1
            candidate_stops = list(veh.route) + [p_stop, d_stop]

            is_feasible, sched_stops, metrics, reason = self.checker.evaluate_candidate_route(
                vehicle=veh,
                candidate_stops=candidate_stops,
                new_request=request,
                active_requests=active_requests,
            )

            if is_feasible:
                # Score based on dispatch time / wait time
                wait_time = metrics["wait_time"]
                ride_time = metrics["in_vehicle_time"]
                journey_time = wait_time + ride_time

                fare = self.fare_calc.compute_fare(
                    direct_distance_km=request.direct_distance_km,
                    direct_time_sec=request.direct_travel_time,
                    actual_time_sec=ride_time,
                    avg_concurrent_passengers=1.0 + (0.5 if len(veh.route) > 0 else 0.0),
                    num_passengers=request.num_passengers,
                )

                score = journey_time + (0.5 * metrics["delay_to_existing"])
                if score < best_score:
                    best_score = score
                    best_decision = AssignmentDecision(
                        vehicle_id=veh.vehicle_id,
                        pickup_index=p_idx,
                        dropoff_index=d_idx,
                        is_feasible=True,
                        delta_vehicle_time=metrics["total_route_time"],
                        delta_vehicle_distance=metrics["total_route_dist"],
                        passenger_wait_time=wait_time,
                        passenger_journey_time=journey_time,
                        passenger_fare=fare,
                        delay_to_existing=metrics["delay_to_existing"],
                        score=score,
                        rejection_reason=None,
                        objective_vector=(journey_time, fare),
                    )

        return best_decision
