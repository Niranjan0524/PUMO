"""Evaluation metrics for paratransit routing algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
from .models import PassengerRequest, RequestStatus, Vehicle


@dataclass
class SimulationMetrics:
    """Comprehensive benchmark metrics for paratransit dispatch algorithms."""
    algorithm_name: str
    total_requests: int
    served_requests: int
    rejected_requests: int
    service_rate_pct: float
    avg_wait_time_sec: float
    avg_wait_time_min: float
    avg_in_vehicle_time_sec: float
    avg_journey_time_sec: float
    avg_journey_time_min: float
    avg_detour_ratio: float
    avg_fare: float
    total_fare_revenue: float
    total_vehicle_distance_km: float
    total_runtime_ms: float
    avg_runtime_ms_per_request: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "total_requests": self.total_requests,
            "served_requests": self.served_requests,
            "rejected_requests": self.rejected_requests,
            "service_rate_pct": round(self.service_rate_pct, 2),
            "avg_wait_time_sec": round(self.avg_wait_time_sec, 2),
            "avg_wait_time_min": round(self.avg_wait_time_min, 2),
            "avg_in_vehicle_time_sec": round(self.avg_in_vehicle_time_sec, 2),
            "avg_journey_time_sec": round(self.avg_journey_time_sec, 2),
            "avg_journey_time_min": round(self.avg_journey_time_min, 2),
            "avg_detour_ratio": round(self.avg_detour_ratio, 3),
            "avg_fare": round(self.avg_fare, 2),
            "total_fare_revenue": round(self.total_fare_revenue, 2),
            "total_vehicle_dist_km": round(self.total_vehicle_distance_km, 2),
            "total_runtime_ms": round(self.total_runtime_ms, 2),
            "avg_runtime_ms": round(self.avg_runtime_ms_per_request, 3),
        }


def compute_simulation_metrics(
    algorithm_name: str,
    requests: List[PassengerRequest],
    vehicles: List[Vehicle],
    runtime_ms: float,
) -> SimulationMetrics:
    """Computes all D-DARP performance metrics from simulation results."""
    total_req = len(requests)
    served = [r for r in requests if r.status == RequestStatus.COMPLETED or (r.pickup_time is not None and r.dropoff_time is not None)]
    rejected = [r for r in requests if r.status == RequestStatus.REJECTED or r.assigned_vehicle_id is None]

    num_served = len(served)
    num_rejected = len(rejected)
    service_rate = (num_served / total_req * 100.0) if total_req > 0 else 0.0

    if num_served > 0:
        avg_wait = sum(r.waiting_time for r in served if r.waiting_time is not None) / num_served
        avg_ride = sum(r.in_vehicle_time for r in served if r.in_vehicle_time is not None) / num_served
        avg_journey = sum(r.total_journey_time for r in served if r.total_journey_time is not None) / num_served
        avg_detour = sum(r.detour_ratio for r in served) / num_served
        avg_fare = sum(r.fare for r in served) / num_served
        total_rev = sum(r.fare for r in served)
    else:
        avg_wait = 0.0
        avg_ride = 0.0
        avg_journey = 0.0
        avg_detour = 1.0
        avg_fare = 0.0
        total_rev = 0.0

    total_dist = sum(v.total_distance_km for v in vehicles)
    avg_runtime_per_req = (runtime_ms / total_req) if total_req > 0 else 0.0

    return SimulationMetrics(
        algorithm_name=algorithm_name,
        total_requests=total_req,
        served_requests=num_served,
        rejected_requests=num_rejected,
        service_rate_pct=service_rate,
        avg_wait_time_sec=avg_wait,
        avg_wait_time_min=avg_wait / 60.0,
        avg_in_vehicle_time_sec=avg_ride,
        avg_journey_time_sec=avg_journey,
        avg_journey_time_min=avg_journey / 60.0,
        avg_detour_ratio=avg_detour,
        avg_fare=avg_fare,
        total_fare_revenue=total_rev,
        total_vehicle_distance_km=total_dist,
        total_runtime_ms=runtime_ms,
        avg_runtime_ms_per_request=avg_runtime_per_req,
    )
