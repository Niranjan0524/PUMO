"""Fare calculation engine for PUMO Paratransit based on Bengaluru benchmarks."""
from __future__ import annotations

from typing import Iterable


class ParatransitFareCalculator:
    """Calculates passenger fares and driver revenue for shared paratransit.

    Implements the pricing model documented in FARE_MODEL.md:
      - Base regulatory meter rate: ₹30 for first 2 km, ₹15/km thereafter.
      - Detour compensation: passengers are discounted for detours.
      - Ride-sharing discount: passengers are discounted when co-riders share the vehicle.
      - Party discount: 40% discount on additional riders in the same group.
      - Price floor: ₹20.0 minimum fare.
    """

    def __init__(
        self,
        base_fare: float = 30.0,
        base_distance_km: float = 2.0,
        per_km_rate: float = 15.0,
        min_fare: float = 20.0,
        max_discount: float = 0.45,
        detour_discount_rate: float = 0.25,
        sharing_discount_rate: float = 0.15,
        group_discount_factor: float = 0.60,
    ):
        self.base_fare = base_fare
        self.base_distance_km = base_distance_km
        self.per_km_rate = per_km_rate
        self.min_fare = min_fare
        self.max_discount = max_discount
        self.detour_discount_rate = detour_discount_rate
        self.sharing_discount_rate = sharing_discount_rate
        self.group_discount_factor = group_discount_factor

    def compute_solo_fare(self, distance_km: float) -> float:
        """Calculates standard regulatory meter fare for a solo ride."""
        if distance_km <= 0.0:
            return self.min_fare
        extra_km = max(0.0, distance_km - self.base_distance_km)
        return max(self.min_fare, self.base_fare + (self.per_km_rate * extra_km))

    def compute_fare(
        self,
        direct_distance_km: float,
        direct_time_sec: float,
        actual_time_sec: float,
        avg_concurrent_passengers: float = 1.0,
        num_passengers: int = 1,
    ) -> float:
        """Calculates discounted shared-auto fare for a completed or simulated trip."""
        solo_fare = self.compute_solo_fare(direct_distance_km)

        # Detour ratio tau >= 1.0
        if direct_time_sec > 0:
            tau = max(1.0, actual_time_sec / direct_time_sec)
        else:
            tau = 1.0

        # Concurrency effect
        co_passengers = max(0.0, avg_concurrent_passengers - 1.0)

        # Total discount fraction
        discount = min(
            self.max_discount,
            (self.detour_discount_rate * (tau - 1.0)) + (self.sharing_discount_rate * co_passengers)
        )
        discount = max(0.0, discount)

        single_rider_fare = max(self.min_fare, solo_fare * (1.0 - discount))

        # Additional riders in party pay discounted rate
        if num_passengers <= 1:
            return round(single_rider_fare, 2)
        else:
            party_fare = single_rider_fare * (1.0 + (self.group_discount_factor * (num_passengers - 1)))
            return round(party_fare, 2)

    def compute_driver_revenue(self, passenger_fares: Iterable[float]) -> float:
        """Total revenue pooled by the driver from all riders on a route."""
        return round(sum(passenger_fares), 2)
