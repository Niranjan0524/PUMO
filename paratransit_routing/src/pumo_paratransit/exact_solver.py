"""Algorithm A7: Exact Optimization Solver for D-DARP benchmark instances."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import itertools
from .models import Stop, StopType, PassengerRequest, Vehicle, RequestStatus
from .dijkstra import DijkstraRouter
from .constraints import RouteConstraintChecker
from .fare import ParatransitFareCalculator


class ExactDARPSolver:
    """Algorithm A7: Exact combinatorial branch-and-bound solver for Dial-a-Ride.

    Solves small-to-medium benchmark instances to proven mathematical global optimality.
    Serves as the ground-truth benchmark to rigorously compute optimality gaps for
    heuristics A2, A3, A4, A5, and A6.
    """

    def __init__(
        self,
        router: DijkstraRouter,
        fare_calc: Optional[ParatransitFareCalculator] = None,
        weight_journey_time: float = 1.0,
        weight_vehicle_dist: float = 0.5,
        penalty_unserved: float = 20000.0,
    ):
        self.router = router
        self.fare_calc = fare_calc or ParatransitFareCalculator()
        self.checker = RouteConstraintChecker(router)
        self.w_journey = weight_journey_time
        self.w_dist = weight_vehicle_dist
        self.penalty_unserved = penalty_unserved

    def solve(
        self,
        requests: List[PassengerRequest],
        vehicles: List[Vehicle],
        current_time: float = 0.0,
    ) -> Dict[str, Any]:
        """Solves the instance to global optimality.

        Returns:
            Dict containing:
              - 'best_cost': Minimum total objective cost
              - 'vehicle_routes': Mapping vehicle_id -> list of scheduled Stops
              - 'served_requests': List of served request IDs
              - 'unserved_requests': List of unserved request IDs
              - 'total_journey_time': Sum of passenger journey times
              - 'total_distance_km': Sum of vehicle travel distances
              - 'optimality_proven': True
        """
        n_req = len(requests)
        n_veh = len(vehicles)
        req_map = {r.request_id: r for r in requests}
        req_ids = list(req_map.keys())

        best_cost = float("inf")
        best_solution: Optional[Dict[str, Any]] = None

        # Pre-build pickup and dropoff stop objects
        stops_by_req: Dict[str, Tuple[Stop, Stop]] = {}
        for r in requests:
            p = Stop(
                stop_id=f"stop_{r.request_id}_p",
                node_id=r.origin_id,
                request_id=r.request_id,
                stop_type=StopType.PICKUP,
                passengers_delta=r.num_passengers,
            )
            d = Stop(
                stop_id=f"stop_{r.request_id}_d",
                node_id=r.destination_id,
                request_id=r.request_id,
                stop_type=StopType.DROPOFF,
                passengers_delta=-r.num_passengers,
            )
            stops_by_req[r.request_id] = (p, d)

        # Iterate over all possible partitions of requests among vehicles and unserved set:
        # Each request r can be assigned to vehicle 0..(n_veh-1), or unserved (-1)
        partition_choices = list(range(n_veh)) + [-1]

        for partition in itertools.product(partition_choices, repeat=n_req):
            # Group requests by assigned vehicle
            veh_requests: Dict[int, List[str]] = {v_idx: [] for v_idx in range(n_veh)}
            unserved_count = 0
            unserved_rids = []

            for r_idx, v_idx in enumerate(partition):
                rid = req_ids[r_idx]
                if v_idx == -1:
                    unserved_count += 1
                    unserved_rids.append(rid)
                else:
                    veh_requests[v_idx].append(rid)

            lower_bound_cost = unserved_count * self.penalty_unserved
            if lower_bound_cost >= best_cost:
                continue

            # For each vehicle, find the best feasible route permutation of its assigned requests
            total_partition_cost = lower_bound_cost
            current_partition_routes: Dict[str, List[Stop]] = {}
            current_partition_journey_times: Dict[str, float] = {}
            current_partition_dist = 0.0
            is_partition_feasible = True

            for v_idx, assigned_rids in veh_requests.items():
                veh = vehicles[v_idx]
                if not assigned_rids:
                    current_partition_routes[veh.vehicle_id] = []
                    continue

                best_veh_cost, best_veh_stops, best_veh_jtimes, best_veh_dist = self._find_best_vehicle_route(
                    vehicle=veh,
                    assigned_rids=assigned_rids,
                    stops_by_req=stops_by_req,
                    req_map=req_map,
                    current_time=current_time,
                )

                if best_veh_cost == float("inf"):
                    is_partition_feasible = False
                    break

                total_partition_cost += best_veh_cost
                current_partition_routes[veh.vehicle_id] = best_veh_stops
                current_partition_journey_times.update(best_veh_jtimes)
                current_partition_dist += best_veh_dist

                if total_partition_cost >= best_cost:
                    is_partition_feasible = False
                    break

            if is_partition_feasible and total_partition_cost < best_cost:
                best_cost = total_partition_cost
                served_rids = [rid for rid in req_ids if rid not in unserved_rids]
                best_solution = {
                    "best_cost": round(best_cost, 2),
                    "vehicle_routes": current_partition_routes,
                    "served_requests": served_rids,
                    "unserved_requests": unserved_rids,
                    "total_journey_time": round(sum(current_partition_journey_times.values()), 2),
                    "total_distance_km": round(current_partition_dist, 2),
                    "optimality_proven": True,
                }

        return best_solution or {
            "best_cost": float("inf"),
            "vehicle_routes": {},
            "served_requests": [],
            "unserved_requests": req_ids,
            "total_journey_time": 0.0,
            "total_distance_km": 0.0,
            "optimality_proven": True,
        }

    def _find_best_vehicle_route(
        self,
        vehicle: Vehicle,
        assigned_rids: List[str],
        stops_by_req: Dict[str, Tuple[Stop, Stop]],
        req_map: Dict[str, PassengerRequest],
        current_time: float,
    ) -> Tuple[float, List[Stop], Dict[str, float], float]:
        """Finds the lowest cost feasible stop ordering for a single vehicle."""
        assigned_stops: List[Stop] = []
        for rid in assigned_rids:
            p, d = stops_by_req[rid]
            assigned_stops.extend([p, d])

        best_cost = float("inf")
        best_stops: List[Stop] = []
        best_journey_times: Dict[str, float] = {}
        best_dist = 0.0

        # Enumerate valid permutations that satisfy precedence
        for perm in itertools.permutations(assigned_stops):
            # Fast precedence check: p must appear before d
            p_pos: Dict[str, int] = {}
            valid_order = True
            for idx, s in enumerate(perm):
                if s.stop_type == StopType.PICKUP:
                    p_pos[s.request_id] = idx
                elif s.stop_type == StopType.DROPOFF:
                    if s.request_id not in p_pos or p_pos[s.request_id] > idx:
                        valid_order = False
                        break
            if not valid_order:
                continue

            # Simulate schedule and check constraints
            is_valid, sched_stops, total_tt, total_dist, _ = self.checker.simulate_schedule(
                start_node=vehicle.current_node,
                start_time=vehicle.current_time,
                initial_load=vehicle.occupied_seats,
                capacity=vehicle.capacity,
                stops=list(perm),
            )

            if not is_valid:
                continue

            # Check time windows for each request
            sched_map: Dict[str, Dict[str, float]] = {}
            for s in sched_stops:
                if s.request_id not in sched_map:
                    sched_map[s.request_id] = {}
                sched_map[s.request_id][s.stop_type.value] = s.planned_arrival_time

            feasible_windows = True
            veh_journey_sum = 0.0
            veh_jtimes: Dict[str, float] = {}

            for rid in assigned_rids:
                req = req_map[rid]
                p_arr = sched_map[rid][StopType.PICKUP.value]
                d_arr = sched_map[rid][StopType.DROPOFF.value]

                wait = p_arr - req.request_time
                ride = d_arr - p_arr
                if wait > req.max_wait_time:
                    feasible_windows = False
                    break
                if ride > (req.direct_travel_time * req.max_detour_ratio):
                    feasible_windows = False
                    break

                j_time = wait + ride
                veh_journey_sum += j_time
                veh_jtimes[rid] = j_time

            if not feasible_windows:
                continue

            cost = (self.w_journey * veh_journey_sum) + (self.w_dist * total_dist)
            if cost < best_cost:
                best_cost = cost
                best_stops = sched_stops
                best_journey_times = veh_jtimes
                best_dist = total_dist

        return best_cost, best_stops, best_journey_times, best_dist
