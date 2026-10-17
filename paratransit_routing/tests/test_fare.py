"""Unit tests for the Paratransit Fare Model."""
from __future__ import annotations

import unittest
import os
import sys

# Add src to sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pumo_paratransit.fare import ParatransitFareCalculator


class TestFareModel(unittest.TestCase):

    def setUp(self):
        self.calc = ParatransitFareCalculator()

    def test_solo_baseline_fare(self):
        # 1.5 km (within base distance 2.0 km) -> base fare ₹30
        self.assertEqual(self.calc.compute_solo_fare(1.5), 30.0)
        # 2.0 km -> base fare ₹30
        self.assertEqual(self.calc.compute_solo_fare(2.0), 30.0)
        # 5.0 km -> 30 + (5 - 2) * 15 = 30 + 45 = ₹75
        self.assertEqual(self.calc.compute_solo_fare(5.0), 75.0)

    def test_shared_discount_with_detour_and_riders(self):
        # Direct trip: 5 km, 600s direct time, actual 750s (detour tau = 1.25), 2 riders sharing
        # Solo fare is 75.0
        # Detour excess = 0.25 -> 0.25 * 0.25 = 0.0625
        # Sharing co-riders = 1.0 -> 0.15 * 1.0 = 0.15
        # Total discount = 0.2125 (21.25%)
        # Expected fare = 75 * (1 - 0.2125) = 75 * 0.7875 = 59.06
        fare = self.calc.compute_fare(
            direct_distance_km=5.0,
            direct_time_sec=600.0,
            actual_time_sec=750.0,
            avg_concurrent_passengers=2.0,
            num_passengers=1,
        )
        self.assertAlmostEqual(fare, 59.06, places=1)
        self.assertLess(fare, 75.0)

    def test_party_discount(self):
        # Party of 2 passengers: second passenger gets 40% discount (factor 0.6)
        single_fare = self.calc.compute_fare(
            direct_distance_km=4.0,
            direct_time_sec=500.0,
            actual_time_sec=500.0,
            avg_concurrent_passengers=1.0,
            num_passengers=1,
        )
        party_fare = self.calc.compute_fare(
            direct_distance_km=4.0,
            direct_time_sec=500.0,
            actual_time_sec=500.0,
            avg_concurrent_passengers=1.0,
            num_passengers=2,
        )
        expected = round(single_fare * 1.6, 2)
        self.assertEqual(party_fare, expected)

    def test_minimum_fare_floor(self):
        fare = self.calc.compute_fare(
            direct_distance_km=0.5,
            direct_time_sec=60.0,
            actual_time_sec=120.0,
            avg_concurrent_passengers=3.0,
            num_passengers=1,
        )
        self.assertGreaterEqual(fare, 20.0)


if __name__ == "__main__":
    unittest.main()
