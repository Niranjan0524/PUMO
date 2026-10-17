"""Algorithm A5: Regret-based Insertion Heuristic (RIH / Regret-k)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from .models import PassengerRequest, Vehicle, AssignmentDecision
from .dijkstra import DijkstraRouter
from .insertion import GreedyInsertionAssigner
from .fare import ParatransitFareCalculator


class RegretInsertionAssigner:
    """Algorithm A5: Regret-k insertion heuristic.

    Computes the difference in cost between the best vehicle assignment and
    the k-th best vehicle assignment. Requests with high regret (at high risk of
    becoming unserved or heavily degraded if not served immediately) are prioritized.
    """

    def __init__(
        self,
        router: DijkstraRouter,
        fare_calc: Optional[ParatransitFareCalculator] = None,
        k: int = 2,
        penalty_unserved: float = 10000.0,
    ):
        self.router = router
        self.fare_calc = fare_calc or ParatransitFareCalculator()
        self.greedy = GreedyInsertionAssigner(router, self.fare_calc)
        self.k = k
        self.penalty_unserved = penalty_unserved

    def calculate_regret_for_request(
        self,
        request: PassengerRequest,
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
    ) -> Tuple[float, Optional[AssignmentDecision]]:
        """Calculates regret-k value and the best assignment decision for a single request."""
        all_insertions = self.greedy.find_all_feasible_insertions(
            request=request,
            vehicles=vehicles,
            active_requests=active_requests,
            current_time=current_time,
        )

        if not all_insertions:
            return -1.0, None

        # Group best insertion by vehicle_id
        best_by_vehicle: Dict[str, AssignmentDecision] = {}
        for ins in all_insertions:
            vid = ins.vehicle_id
            if vid not in best_by_vehicle or ins.score < best_by_vehicle[vid].score:
                best_by_vehicle[vid] = ins

        # Sort vehicle costs in ascending order
        sorted_decisions = sorted(best_by_vehicle.values(), key=lambda d: d.score)
        best_decision = sorted_decisions[0]

        if len(sorted_decisions) == 1:
            # Only one vehicle can serve this request: maximum regret
            regret = self.penalty_unserved
        else:
            # Sum differences for up to k vehicles
            regret = 0.0
            c1 = sorted_decisions[0].score
            limit = min(self.k, len(sorted_decisions))
            for i in range(1, limit):
                regret += (sorted_decisions[i].score - c1)

        return regret, best_decision

    def assign(
        self,
        request: PassengerRequest,
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
    ) -> AssignmentDecision:
        """Assigns an individual dynamic request using regret evaluation."""
        regret, best_decision = self.calculate_regret_for_request(
            request=request,
            vehicles=vehicles,
            active_requests=active_requests,
            current_time=current_time,
        )

        if best_decision is None:
            return AssignmentDecision(
                is_feasible=False,
                rejection_reason="No feasible insertion found (Regret heuristic)",
            )

        return best_decision

    def assign_batch(
        self,
        pending_requests: List[PassengerRequest],
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
    ) -> Dict[str, AssignmentDecision]:
        """Dispatches a batch of pending requests using iterative Regret-k selection."""
        assignments: Dict[str, AssignmentDecision] = {}
        unassigned = list(pending_requests)

        # Working copies of vehicle states
        current_vehicles = [v.copy() for v in vehicles]
        vehicle_map = {v.vehicle_id: v for v in current_vehicles}
        current_active = dict(active_requests)

        while unassigned:
            best_req = None
            max_regret = -float("inf")
            best_req_decision = None

            for req in unassigned:
                regret, decision = self.calculate_regret_for_request(
                    request=req,
                    vehicles=current_vehicles,
                    active_requests=current_active,
                    current_time=current_time,
                )

                if decision is None:
                    continue

                if regret > max_regret:
                    max_regret = regret
                    best_req = req
                    best_req_decision = decision

            if best_req is None or best_req_decision is None:
                # None of the remaining unassigned requests can be served feasibly
                for req in unassigned:
                    assignments[req.request_id] = AssignmentDecision(
                        is_feasible=False,
                        rejection_reason="Infeasible under vehicle capacity / time windows",
                    )
                break

            # Assign best_req to its chosen vehicle
            assignments[best_req.request_id] = best_req_decision
            unassigned.remove(best_req)

            # Update working vehicle route
            vid = best_req_decision.vehicle_id
            veh = vehicle_map[vid]
            from .models import Stop, StopType
            p_stop = Stop(
                stop_id=f"stop_{best_req.request_id}_p",
                node_id=best_req.origin_id,
                request_id=best_req.request_id,
                stop_type=StopType.PICKUP,
                passengers_delta=best_req.num_passengers,
            )
            d_stop = Stop(
                stop_id=f"stop_{best_req.request_id}_d",
                node_id=best_req.destination_id,
                request_id=best_req.request_id,
                stop_type=StopType.DROPOFF,
                passengers_delta=-best_req.num_passengers,
            )

            p_idx = best_req_decision.pickup_index
            d_idx = best_req_decision.dropoff_index
            veh.route.insert(p_idx, p_stop)
            veh.route.insert(d_idx, d_stop)

            # Re-simulate schedule on vehicle to update stop arrival/departure times
            _, sched_stops, _, _, _ = self.greedy.checker.simulate_schedule(
                start_node=veh.current_node,
                start_time=veh.current_time,
                initial_load=veh.occupied_seats,
                capacity=veh.capacity,
                stops=veh.route,
            )
            veh.route = sched_stops
            current_active[best_req.request_id] = best_req

        return assignments
