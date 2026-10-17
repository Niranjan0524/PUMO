"""Algorithm A6: Pareto-based Feasible Insertion (Multi-Objective Optimization)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Literal
import math
from .models import PassengerRequest, Vehicle, AssignmentDecision
from .dijkstra import DijkstraRouter
from .insertion import GreedyInsertionAssigner
from .fare import ParatransitFareCalculator


def dominates(a: Tuple[float, float], b: Tuple[float, float], eps: float = 1e-6) -> bool:
    """Returns True if objective vector 'a' weakly improves 'b' and strictly improves >= 1 component."""
    return (a[0] <= b[0] + eps and a[1] <= b[1] + eps) and (a[0] < b[0] - eps or a[1] < b[1] - eps)


class ParetoMultiObjectiveAssigner:
    """Algorithm A6: Pareto-based feasible insertion optimizing Journey Time and Fare.

    Rather than collapsing time and fare into an arbitrary fixed scalar weight,
    A6 evaluates the 2D objective space (Passenger Journey Time, Passenger Fare),
    prunes dominated insertions, and offers explicit Pareto trade-off choices:
      - 'time_first': Minimizes journey time (fastest travel).
      - 'fare_first': Minimizes fare paid (maximum shared discount).
      - 'balanced': Compromise knee-point closest to the normalized ideal point.
    """

    def __init__(
        self,
        router: DijkstraRouter,
        fare_calc: Optional[ParatransitFareCalculator] = None,
        strategy: Literal["time_first", "fare_first", "balanced"] = "balanced",
    ):
        self.router = router
        self.fare_calc = fare_calc or ParatransitFareCalculator()
        self.greedy = GreedyInsertionAssigner(router, self.fare_calc)
        self.strategy = strategy

    def compute_pareto_front(
        self,
        request: PassengerRequest,
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
    ) -> List[AssignmentDecision]:
        """Finds all non-dominated feasible assignment decisions in (JourneyTime, Fare) space."""
        candidates = self.greedy.find_all_feasible_insertions(
            request=request,
            vehicles=vehicles,
            active_requests=active_requests,
            current_time=current_time,
        )

        if not candidates:
            return []

        # Extract objective vectors: (passenger_journey_time, passenger_fare)
        pareto_decisions: List[AssignmentDecision] = []

        for i, cand in enumerate(candidates):
            obj_cand = (cand.passenger_journey_time, cand.passenger_fare)
            is_dominated = False

            for j, other in enumerate(candidates):
                if i == j:
                    continue
                obj_other = (other.passenger_journey_time, other.passenger_fare)
                if dominates(obj_other, obj_cand):
                    is_dominated = True
                    break

            if not is_dominated:
                # Retain only unique points
                if not any(
                    math.isclose(cand.passenger_journey_time, p.passenger_journey_time, abs_tol=1e-3)
                    and math.isclose(cand.passenger_fare, p.passenger_fare, abs_tol=1e-3)
                    for p in pareto_decisions
                ):
                    pareto_decisions.append(cand)

        # Sort front by journey time ascending
        pareto_decisions.sort(key=lambda d: d.passenger_journey_time)
        return pareto_decisions

    def select_from_pareto_front(
        self,
        pareto_front: List[AssignmentDecision],
        strategy: Optional[Literal["time_first", "fare_first", "balanced"]] = None,
    ) -> Optional[AssignmentDecision]:
        """Selects a specific assignment decision from the Pareto frontier."""
        if not pareto_front:
            return None

        strat = strategy or self.strategy

        if strat == "time_first":
            # Minimum journey time
            return min(pareto_front, key=lambda d: d.passenger_journey_time)

        elif strat == "fare_first":
            # Minimum fare paid
            return min(pareto_front, key=lambda d: d.passenger_fare)

        elif strat == "balanced":
            if len(pareto_front) == 1:
                return pareto_front[0]

            min_time = min(d.passenger_journey_time for d in pareto_front)
            max_time = max(d.passenger_journey_time for d in pareto_front)
            min_fare = min(d.passenger_fare for d in pareto_front)
            max_fare = max(d.passenger_fare for d in pareto_front)

            time_range = max(1.0, max_time - min_time)
            fare_range = max(1.0, max_fare - min_fare)

            # Find knee-point minimizing Euclidean distance to ideal point (min_time, min_fare)
            def normalized_distance(d: AssignmentDecision) -> float:
                norm_t = (d.passenger_journey_time - min_time) / time_range
                norm_f = (d.passenger_fare - min_fare) / fare_range
                return math.sqrt(norm_t ** 2 + norm_f ** 2)

            return min(pareto_front, key=normalized_distance)

        else:
            raise ValueError(f"Unknown Pareto selection strategy: {strat}")

    def assign(
        self,
        request: PassengerRequest,
        vehicles: List[Vehicle],
        active_requests: Dict[str, PassengerRequest],
        current_time: float,
    ) -> AssignmentDecision:
        """Assigns the request using the chosen Pareto multi-objective strategy."""
        pareto_front = self.compute_pareto_front(
            request=request,
            vehicles=vehicles,
            active_requests=active_requests,
            current_time=current_time,
        )

        if not pareto_front:
            return AssignmentDecision(
                is_feasible=False,
                rejection_reason="No feasible non-dominated insertion found",
            )

        chosen = self.select_from_pareto_front(pareto_front, self.strategy)
        if chosen is None:
            return AssignmentDecision(is_feasible=False, rejection_reason="Pareto selection returned empty")
        return chosen
