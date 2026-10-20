"""
Preference-Weighted Assignment Algorithm
-----------------------------------------
Replaces ad hoc scalar weights with empirically grounded WTP weights from:
  - Das, Bhaduri & Velaga (2023) — Paper 2 Mixed Logit coefficients
  - Das & Mandal (2021) — Paper 3 Crowding penalty

The assigner inherits all feasibility checking from GreedyInsertionAssigner
but scores candidates using generalised cost in INR (a passenger-utility-grounded
composite metric) instead of arbitrary weights.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from .models import PassengerRequest, Vehicle, AssignmentDecision
from .dijkstra import DijkstraRouter
from .insertion import GreedyInsertionAssigner
from .fare import ParatransitFareCalculator
from .passenger_preference import PassengerUtility, PassengerSegment, crowding_discomfort


class PreferenceWeightedAssigner:
    """Assignment algorithm whose scoring uses WTP-calibrated generalised cost.

    Difference from A4 (GreedyInsertionAssigner):
      - The cost function is replaced by generalised_cost_inr from the
        PassengerUtility model (Papers 2 & 3).
      - Crowding above 75% capacity incurs an explicit disutility penalty.
      - Real-time information availability reduces perceived generalised cost.
      - Segment-specific coefficients adjust for passenger heterogeneity.

    Usage: instantiate with a passenger segment to evaluate assignments from
    the perspective of that segment's preferences.
    """

    def __init__(
        self,
        router: DijkstraRouter,
        fare_calc: Optional[ParatransitFareCalculator] = None,
        segment: PassengerSegment = PassengerSegment.OCCASIONAL_USER,
        has_realtime_info: bool = True,
        preference_model: Optional[PassengerUtility] = None,
    ):
        self.router = router
        self.fare_calc = fare_calc or ParatransitFareCalculator()
        self.segment = segment
        self.has_realtime_info = has_realtime_info
        self.preference = preference_model or PassengerUtility()
        self.greedy = GreedyInsertionAssigner(router, self.fare_calc)

    def assign(
        self,
        request: PassengerRequest,
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
        access_time_min: float = 2.0,
    ) -> AssignmentDecision:
        """Assigns the request using preference-calibrated generalised cost scoring.

        Access time (walk to pickup) defaults to 2 minutes unless the caller
        supplies a GPS-derived estimate.
        """
        # Get all feasible insertions (reuse A4's enumeration + constraint checking)
        candidates = self.greedy.find_all_feasible_insertions(
            request=request,
            vehicles=vehicles,
            active_requests=active_requests,
            current_time=current_time,
        )

        if not candidates:
            return AssignmentDecision(
                is_feasible=False,
                rejection_reason=(
                    "No feasible insertion found "
                    f"(PreferenceWeighted, segment={self.segment.value})"
                ),
            )

        best_gc = float("inf")
        best_decision: Optional[AssignmentDecision] = None

        for cand in candidates:
            # Compute occupancy ratio for the candidate vehicle at pickup moment
            veh = next((v for v in vehicles if v.vehicle_id == cand.vehicle_id), None)
            if veh is None:
                continue

            # Passengers on board at time of new request pickup
            occupancy_at_pickup = (veh.occupied_seats + request.num_passengers) / max(1, veh.capacity)

            wait_min = cand.passenger_wait_time / 60.0
            ride_min = cand.passenger_journey_time / 60.0 - wait_min

            # Use preference-calibrated generalised cost
            gc = self.preference.compute_generalised_cost_inr(
                in_vehicle_time_min=ride_min,
                waiting_time_min=wait_min,
                access_time_min=access_time_min,
                fare_inr=cand.passenger_fare,
                occupancy_ratio=occupancy_at_pickup,
                has_realtime_info=self.has_realtime_info,
                segment=self.segment,
            )

            if gc < best_gc:
                best_gc = gc
                # Annotate score with the generalised cost (in INR)
                best_decision = AssignmentDecision(
                    vehicle_id=cand.vehicle_id,
                    pickup_index=cand.pickup_index,
                    dropoff_index=cand.dropoff_index,
                    is_feasible=True,
                    delta_vehicle_time=cand.delta_vehicle_time,
                    delta_vehicle_distance=cand.delta_vehicle_distance,
                    passenger_wait_time=cand.passenger_wait_time,
                    passenger_journey_time=cand.passenger_journey_time,
                    passenger_fare=cand.passenger_fare,
                    delay_to_existing=cand.delay_to_existing,
                    score=gc,
                    rejection_reason=None,
                    objective_vector=(cand.passenger_journey_time, cand.passenger_fare, gc),
                )

        return best_decision or AssignmentDecision(is_feasible=False, rejection_reason="All candidates scored infinite GC")
