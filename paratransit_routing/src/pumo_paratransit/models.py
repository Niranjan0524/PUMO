"""Data models for paratransit routing, requests, vehicles, and stops."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any


class StopType(str, Enum):
    """Type of vehicle stop."""
    PICKUP = "PICKUP"
    DROPOFF = "DROPOFF"


class RequestStatus(str, Enum):
    """Lifecycle status of a passenger trip request."""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_TRANSIT = "IN_TRANSIT"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"


@dataclass
class Stop:
    """A scheduled stop on a shared auto's route."""
    stop_id: str
    node_id: str
    request_id: str
    stop_type: StopType
    passengers_delta: int
    planned_arrival_time: float = 0.0
    planned_departure_time: float = 0.0
    service_duration: float = 30.0  # seconds for boarding/alighting

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop_id": self.stop_id,
            "node_id": self.node_id,
            "request_id": self.request_id,
            "stop_type": self.stop_type.value,
            "passengers_delta": self.passengers_delta,
            "planned_arrival_time": round(self.planned_arrival_time, 2),
            "planned_departure_time": round(self.planned_departure_time, 2),
            "service_duration": self.service_duration,
        }


@dataclass
class PassengerRequest:
    """Passenger trip request in dynamic paratransit."""
    request_id: str
    origin_id: str
    destination_id: str
    request_time: float  # seconds from simulation start
    num_passengers: int = 1
    max_wait_time: float = 600.0  # seconds (10 minutes)
    max_detour_ratio: float = 1.6  # max allowable ride time = direct_time * ratio
    status: RequestStatus = RequestStatus.PENDING

    # Outcome tracking
    assigned_vehicle_id: Optional[str] = None
    pickup_time: Optional[float] = None
    dropoff_time: Optional[float] = None
    fare: float = 0.0
    direct_travel_time: float = 0.0
    direct_distance_km: float = 0.0
    rejection_reason: Optional[str] = None

    @property
    def pickup_deadline(self) -> float:
        """Latest allowable pickup arrival time."""
        return self.request_time + self.max_wait_time

    @property
    def dropoff_deadline(self) -> float:
        """Latest allowable dropoff time once picked up."""
        pickup_reference = self.pickup_time if self.pickup_time is not None else self.pickup_deadline
        return pickup_reference + (self.direct_travel_time * self.max_detour_ratio)

    @property
    def waiting_time(self) -> Optional[float]:
        if self.pickup_time is not None:
            return max(0.0, self.pickup_time - self.request_time)
        return None

    @property
    def in_vehicle_time(self) -> Optional[float]:
        if self.pickup_time is not None and self.dropoff_time is not None:
            return max(0.0, self.dropoff_time - self.pickup_time)
        return None

    @property
    def total_journey_time(self) -> Optional[float]:
        if self.dropoff_time is not None:
            return max(0.0, self.dropoff_time - self.request_time)
        return None

    @property
    def detour_ratio(self) -> float:
        if self.in_vehicle_time is not None and self.direct_travel_time > 0:
            return round(self.in_vehicle_time / self.direct_travel_time, 3)
        return 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "origin_id": self.origin_id,
            "destination_id": self.destination_id,
            "request_time": self.request_time,
            "num_passengers": self.num_passengers,
            "max_wait_time": self.max_wait_time,
            "max_detour_ratio": self.max_detour_ratio,
            "status": self.status.value,
            "assigned_vehicle_id": self.assigned_vehicle_id,
            "pickup_time": round(self.pickup_time, 2) if self.pickup_time is not None else None,
            "dropoff_time": round(self.dropoff_time, 2) if self.dropoff_time is not None else None,
            "waiting_time": round(self.waiting_time, 2) if self.waiting_time is not None else None,
            "in_vehicle_time": round(self.in_vehicle_time, 2) if self.in_vehicle_time is not None else None,
            "total_journey_time": round(self.total_journey_time, 2) if self.total_journey_time is not None else None,
            "detour_ratio": self.detour_ratio,
            "fare": round(self.fare, 2),
            "direct_travel_time": round(self.direct_travel_time, 2),
            "direct_distance_km": round(self.direct_distance_km, 2),
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class Vehicle:
    """Shared auto / paratransit vehicle."""
    vehicle_id: str
    capacity: int = 4
    current_node: str = ""
    current_time: float = 0.0
    occupied_seats: int = 0
    route: List[Stop] = field(default_factory=list)
    total_distance_km: float = 0.0
    total_travel_time_sec: float = 0.0
    served_requests_count: int = 0

    @property
    def is_idle(self) -> bool:
        """Vehicle is idle if no stops remain and no passengers are onboard."""
        return len(self.route) == 0 and self.occupied_seats == 0

    @property
    def available_capacity(self) -> int:
        return max(0, self.capacity - self.occupied_seats)

    def copy(self) -> Vehicle:
        """Return a deep copy of vehicle state for hypothetical route evaluations."""
        import copy
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "capacity": self.capacity,
            "current_node": self.current_node,
            "current_time": round(self.current_time, 2),
            "occupied_seats": self.occupied_seats,
            "route_length": len(self.route),
            "total_distance_km": round(self.total_distance_km, 2),
            "total_travel_time_sec": round(self.total_travel_time_sec, 2),
            "served_requests_count": self.served_requests_count,
        }


@dataclass
class AssignmentDecision:
    """Outcome of attempting to assign a request to a vehicle."""
    vehicle_id: Optional[str] = None
    pickup_index: int = -1
    dropoff_index: int = -1
    is_feasible: bool = False
    delta_vehicle_time: float = float("inf")
    delta_vehicle_distance: float = float("inf")
    passenger_wait_time: float = float("inf")
    passenger_journey_time: float = float("inf")
    passenger_fare: float = 0.0
    delay_to_existing: float = 0.0
    score: float = float("inf")
    rejection_reason: Optional[str] = None
    objective_vector: Tuple[float, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> bool:
        return self.is_feasible and self.vehicle_id is not None
