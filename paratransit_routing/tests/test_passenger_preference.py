"""Unit tests for Papers 2 & 3 — Passenger Preference Model."""
from __future__ import annotations
import os
import sys
import unittest

test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pumo_paratransit.passenger_preference import (
    MixedLogitCoefficients,
    PassengerUtility,
    PassengerSegment,
    crowding_discomfort,
)


class TestCrowdingDiscomfort(unittest.TestCase):
    """Tests Paper 3 (Das & Mandal 2021) crowding penalty model."""

    def test_no_penalty_below_threshold(self):
        # Below 75% occupancy → zero discomfort (Paper 3: threshold is 0.75)
        self.assertAlmostEqual(crowding_discomfort(0.0), 0.0)
        self.assertAlmostEqual(crowding_discomfort(0.50), 0.0)
        self.assertAlmostEqual(crowding_discomfort(0.75), 0.0)

    def test_penalty_above_threshold(self):
        # Above 75% → positive crowding discomfort
        penalty_80 = crowding_discomfort(0.80)
        penalty_100 = crowding_discomfort(1.00)
        self.assertGreater(penalty_80, 0.0)
        self.assertGreater(penalty_100, penalty_80)

    def test_full_crowding_bounded_at_one(self):
        self.assertAlmostEqual(crowding_discomfort(1.0), 1.0, places=3)


class TestMixedLogitCoefficients(unittest.TestCase):
    """Tests Paper 2 (Das et al. 2023) WTP estimation."""

    def setUp(self):
        self.coeff = MixedLogitCoefficients()

    def test_wtp_waiting_time_exceeds_travel_time(self):
        """Paper 2 key finding: waiting time WTP > travel time WTP."""
        wtp_wt = self.coeff.wtp_waiting_time_per_min
        wtp_tt = self.coeff.wtp_travel_time_per_min
        self.assertGreater(wtp_wt, wtp_tt,
            msg="Waiting time must be valued higher than in-vehicle time (Paper 2 finding)")

    def test_wtp_access_time_exceeds_travel_time(self):
        """Paper 2 key finding: access/walk time WTP > travel time WTP."""
        wtp_at = self.coeff.wtp_access_time_per_min
        wtp_tt = self.coeff.wtp_travel_time_per_min
        self.assertGreater(wtp_at, wtp_tt,
            msg="Access time must be valued higher than in-vehicle time (Paper 2 finding)")

    def test_wtp_values_are_positive(self):
        self.assertGreater(self.coeff.wtp_travel_time_per_min, 0.0)
        self.assertGreater(self.coeff.wtp_waiting_time_per_min, 0.0)
        self.assertGreater(self.coeff.wtp_access_time_per_min, 0.0)


class TestPassengerUtility(unittest.TestCase):
    """Tests the Mixed Logit utility calculator."""

    def setUp(self):
        self.utility_model = PassengerUtility()

    def test_higher_wait_reduces_utility(self):
        """Utility must fall as waiting time increases."""
        u_low_wait = self.utility_model.compute_utility(
            in_vehicle_time_min=20, waiting_time_min=3, access_time_min=2,
            fare_inr=50.0, occupancy_ratio=0.5, segment=PassengerSegment.OCCASIONAL_USER)
        u_high_wait = self.utility_model.compute_utility(
            in_vehicle_time_min=20, waiting_time_min=12, access_time_min=2,
            fare_inr=50.0, occupancy_ratio=0.5, segment=PassengerSegment.OCCASIONAL_USER)
        self.assertGreater(u_low_wait, u_high_wait)

    def test_overcrowding_reduces_utility(self):
        """Occupancy above 75% threshold must reduce utility (Paper 3)."""
        u_low_crowd = self.utility_model.compute_utility(
            in_vehicle_time_min=20, waiting_time_min=5, access_time_min=2,
            fare_inr=50.0, occupancy_ratio=0.50, segment=PassengerSegment.OCCASIONAL_USER)
        u_high_crowd = self.utility_model.compute_utility(
            in_vehicle_time_min=20, waiting_time_min=5, access_time_min=2,
            fare_inr=50.0, occupancy_ratio=1.00, segment=PassengerSegment.OCCASIONAL_USER)
        self.assertGreater(u_low_crowd, u_high_crowd)

    def test_realtime_info_increases_utility(self):
        """Real-time information availability must increase utility (Paper 2 finding)."""
        u_no_info = self.utility_model.compute_utility(
            in_vehicle_time_min=20, waiting_time_min=5, access_time_min=2,
            fare_inr=50.0, occupancy_ratio=0.5, has_realtime_info=False,
            segment=PassengerSegment.OCCASIONAL_USER)
        u_with_info = self.utility_model.compute_utility(
            in_vehicle_time_min=20, waiting_time_min=5, access_time_min=2,
            fare_inr=50.0, occupancy_ratio=0.5, has_realtime_info=True,
            segment=PassengerSegment.OCCASIONAL_USER)
        self.assertGreater(u_with_info, u_no_info)

    def test_segment_heterogeneity(self):
        """Time-sensitive passengers must assign higher disutility to long waits."""
        kwargs = dict(in_vehicle_time_min=20, waiting_time_min=10, access_time_min=2,
                      fare_inr=50.0, occupancy_ratio=0.5)
        u_time_sens = self.utility_model.compute_utility(
            **kwargs, segment=PassengerSegment.TIME_SENSITIVE)
        u_regular = self.utility_model.compute_utility(
            **kwargs, segment=PassengerSegment.REGULAR_COMMUTER)
        # Time-sensitive passengers should find the same wait more disutile
        self.assertLess(u_time_sens, u_regular)

    def test_generalised_cost_is_non_negative(self):
        gc = self.utility_model.compute_generalised_cost_inr(
            in_vehicle_time_min=25, waiting_time_min=5, access_time_min=3,
            fare_inr=60.0, occupancy_ratio=0.6, has_realtime_info=True,
            segment=PassengerSegment.REGULAR_COMMUTER)
        self.assertGreaterEqual(gc, 0.0)

    def test_wtp_summary_format(self):
        summary = self.utility_model.wtp_summary(PassengerSegment.TIME_SENSITIVE)
        self.assertIn("wtp_waiting_time_per_min_inr", summary)
        self.assertGreater(summary["wtp_waiting_time_per_min_inr"], 0.0)


if __name__ == "__main__":
    unittest.main()
