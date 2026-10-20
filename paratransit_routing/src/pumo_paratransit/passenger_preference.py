"""
Papers 2 & 3 — Passenger Preference Model
------------------------------------------
Mixed Logit utility model and willingness-to-pay estimates for shared-auto choice.

What we take from each paper:

Paper 2 — Das, Bhaduri & Velaga (2023), Transport Policy 143:132-149
  - Stated preference (SP) experiment with 5 service attributes.
  - Mixed Logit model capturing preference heterogeneity.
  - KEY FINDING: travel time reliability and access time are the most critical
    attributes for both users and non-users in Indian urban auto-rickshaw sharing.
  - WTP estimated as β_attribute / (-β_cost) in monetary terms.
  - Female passengers and regular users have higher WTP for real-time information.

Paper 3 — Das & Mandal (2021), Case Studies on Transport Policy:
  - 503 respondents, Mumbai Metropolitan Region.
  - Non-users penalize safety/driver behaviour heavily (key deterrent).
  - Regular users tolerate mild overcrowding if cost savings are significant.
  - HIGH OCCUPANCY (>75% capacity) significantly reduces willingness to travel.
  - Waiting time acts as a stronger deterrent than equivalent in-vehicle time.

Implementation:
  1. Mixed Logit utility function U(option) = Σ β_k * x_k  (Paper 2 structure)
  2. WTP-calibrated attribute weights (ratio β_k / β_cost)   (Paper 2 estimates)
  3. Crowding penalty function above 75% occupancy threshold  (Paper 3 finding)
  4. Passenger segment heterogeneity: Regular, Occasional, Female, Time-Sensitive (Paper 2 & 3)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Passenger segments (Das et al. 2023 & Das & Mandal 2021 segmentation)
# ---------------------------------------------------------------------------

class PassengerSegment(str, Enum):
    """Passenger segments from Paper 2 & 3 preference experiments.

    Each segment carries different β coefficient magnitudes, reflecting
    revealed preference heterogeneity in the Mixed Logit estimates.
    """
    REGULAR_COMMUTER = "regular_commuter"       # Daily users: cost-sensitive
    OCCASIONAL_USER = "occasional_user"          # Infrequent: moderate sensitivity
    TIME_SENSITIVE = "time_sensitive"            # Prioritize travel time over fare
    SAFETY_CONSCIOUS = "safety_conscious"        # Heavy deterrent from overcrowding/driver behavior (non-users)


# ---------------------------------------------------------------------------
# Mixed Logit Utility Specification  (Paper 2 — Das, Bhaduri & Velaga 2023)
# ---------------------------------------------------------------------------

@dataclass
class MixedLogitCoefficients:
    """Utility-function β coefficients calibrated from Das et al. (2023) Table results.

    All time attributes have negative coefficients (disutility increases with time).
    The fare coefficient is negative (passengers prefer lower cost).

    WTP_k = β_k / (-β_fare)  gives monetary value per unit of attribute k.

    Calibrated with reference to the WTP ranges reported in the paper
    and normalised to INR/minute as the monetary unit:
      - Waiting time WTP ≈ 1.5–2.5× in-vehicle time WTP (consistent with literature)
      - Access time WTP ≈ 1.2–2.0× in-vehicle time WTP
      - Reliability WTP: highest among all time attributes (Paper 2 key finding)

    Units:
      β_travel_time   : utility / minute of in-vehicle travel
      β_waiting_time  : utility / minute of waiting at stop or roadside
      β_access_time   : utility / minute of walking to the pickup point
      β_fare          : utility / INR of fare paid
      β_crowding      : utility / crowding level (0–1 scale)
      β_realtime_info : utility / presence of real-time information (binary: 0 or 1)
    """
    # Point estimates derived from Paper 2 Mixed Logit model
    beta_travel_time: float = -0.045      # per minute of in-vehicle time
    beta_waiting_time: float = -0.085     # per minute waiting (≈ 1.9× travel time β)
    beta_access_time: float = -0.065      # per minute walking to pickup (≈ 1.4× travel time β)
    beta_fare: float = -0.030             # per INR of fare
    beta_crowding: float = -0.250         # per unit occupancy ratio > 0.75 (Paper 3)
    beta_realtime_info: float = 0.120     # presence of real-time trip info (positive utility)

    # Standard deviations of random parameters in the Mixed Logit
    # (captures preference heterogeneity across passengers — Paper 2 key finding)
    sigma_travel_time: float = 0.015
    sigma_waiting_time: float = 0.025
    sigma_access_time: float = 0.020

    @property
    def wtp_travel_time_per_min(self) -> float:
        """WTP for 1 minute reduction in in-vehicle travel time (INR/min)."""
        return abs(self.beta_travel_time / self.beta_fare)

    @property
    def wtp_waiting_time_per_min(self) -> float:
        """WTP for 1 minute reduction in waiting time (INR/min)."""
        return abs(self.beta_waiting_time / self.beta_fare)

    @property
    def wtp_access_time_per_min(self) -> float:
        """WTP for 1 minute reduction in access/walk time to pickup (INR/min)."""
        return abs(self.beta_access_time / self.beta_fare)

    @property
    def wtp_real_time_info(self) -> float:
        """Monetary WTP for availability of real-time trip information (INR per trip)."""
        return abs(self.beta_realtime_info / self.beta_fare)


# Segment-specific coefficient adjustments (multiplicative scale factors)
SEGMENT_COEFFICIENT_SCALES: Dict[PassengerSegment, Dict[str, float]] = {
    PassengerSegment.REGULAR_COMMUTER: {
        "travel_time": 0.85,    # Regular commuters accept more in-vehicle time (habituated)
        "waiting_time": 1.20,   # But are more sensitive to waiting (time-constrained commuters)
        "access_time": 1.10,
        "fare": 1.40,           # Most price-sensitive segment
        "crowding": 0.80,       # More tolerant of crowding (accustomed to it)
        "realtime_info": 1.20,
    },
    PassengerSegment.OCCASIONAL_USER: {
        "travel_time": 1.00,
        "waiting_time": 1.00,
        "access_time": 1.00,
        "fare": 1.00,
        "crowding": 1.20,       # Less accustomed to crowding
        "realtime_info": 1.00,
    },
    PassengerSegment.TIME_SENSITIVE: {
        "travel_time": 1.60,    # Strongly dislikes long in-vehicle time
        "waiting_time": 1.80,   # Very high waiting time aversion
        "access_time": 1.50,
        "fare": 0.60,           # Willing to pay more for speed
        "crowding": 1.10,
        "realtime_info": 1.40,
    },
    PassengerSegment.SAFETY_CONSCIOUS: {
        "travel_time": 1.00,
        "waiting_time": 1.10,
        "access_time": 1.00,
        "fare": 0.90,
        "crowding": 1.80,       # Heavy crowding penalty (Paper 3: safety/comfort deterrent)
        "realtime_info": 1.60,  # High WTP for real-time info (security benefit)
    },
}


# ---------------------------------------------------------------------------
# Crowding Discomfort Model  (Paper 3 — Das & Mandal 2021)
# ---------------------------------------------------------------------------

def crowding_discomfort(occupancy_ratio: float, threshold: float = 0.75) -> float:
    """Computes the crowding discomfort level.

    Paper 3 finding: Comfort deteriorates noticeably above 75% capacity.
    Below the threshold the crowding penalty is zero.
    Above the threshold, discomfort rises as a quadratic function of excess occupancy.

    Args:
        occupancy_ratio: fraction of vehicle capacity currently occupied (0.0–1.0)
        threshold: occupancy ratio above which crowding discomfort begins (Paper 3: 0.75)

    Returns:
        Crowding discomfort value in [0, 1] where 1 = fully packed.
    """
    if occupancy_ratio <= threshold:
        return 0.0
    excess = (occupancy_ratio - threshold) / (1.0 - threshold)
    return min(1.0, excess ** 2)


# ---------------------------------------------------------------------------
# Passenger Utility Calculator
# ---------------------------------------------------------------------------

@dataclass
class PassengerUtility:
    """Computes mixed-logit utility for a shared-auto service option.

    Uses coefficients from Paper 2 (Das, Bhaduri & Velaga 2023) and the
    crowding model from Paper 3 (Das & Mandal 2021).
    """
    base_coefficients: MixedLogitCoefficients = field(
        default_factory=MixedLogitCoefficients
    )
    seed: int = 42

    def draw_individual_coefficients(
        self,
        segment: PassengerSegment,
        individual_seed: Optional[int] = None,
    ) -> MixedLogitCoefficients:
        """Draws individual-level coefficients from the Mixed Logit distribution.

        In a full Mixed Logit model, each individual has their own draw from
        the mixing distribution. Here we simulate this with Normal random draws
        around the segment-specific mean.
        """
        rng = random.Random(individual_seed or self.seed)
        scales = SEGMENT_COEFFICIENT_SCALES[segment]
        bc = self.base_coefficients

        def draw(mean: float, sigma: float) -> float:
            return mean + rng.gauss(0.0, sigma)

        return MixedLogitCoefficients(
            beta_travel_time=draw(bc.beta_travel_time * scales["travel_time"], bc.sigma_travel_time),
            beta_waiting_time=draw(bc.beta_waiting_time * scales["waiting_time"], bc.sigma_waiting_time),
            beta_access_time=draw(bc.beta_access_time * scales["access_time"], bc.sigma_access_time),
            beta_fare=bc.beta_fare * scales["fare"],
            beta_crowding=bc.beta_crowding * scales["crowding"],
            beta_realtime_info=bc.beta_realtime_info * scales["realtime_info"],
        )

    def compute_utility(
        self,
        in_vehicle_time_min: float,
        waiting_time_min: float,
        access_time_min: float,
        fare_inr: float,
        occupancy_ratio: float,
        has_realtime_info: bool = False,
        segment: PassengerSegment = PassengerSegment.OCCASIONAL_USER,
        individual_seed: Optional[int] = None,
    ) -> float:
        """Computes the systematic utility V of a shared-auto assignment option.

        Formula (Paper 2, Mixed Logit specification):
            V = β_tt × TT + β_wt × WT + β_at × AT
                + β_f × Fare + β_cr × Crowding + β_rt × RealTimeInfo

        All time inputs must be in minutes. Fare in INR.

        Returns:
            Systematic utility V (higher = more preferred).
        """
        coeff = self.draw_individual_coefficients(segment, individual_seed)

        crowding = crowding_discomfort(occupancy_ratio)
        real_time_val = 1.0 if has_realtime_info else 0.0

        V = (
            coeff.beta_travel_time * in_vehicle_time_min
            + coeff.beta_waiting_time * waiting_time_min
            + coeff.beta_access_time * access_time_min
            + coeff.beta_fare * fare_inr
            + coeff.beta_crowding * crowding
            + coeff.beta_realtime_info * real_time_val
        )
        return V

    def compute_generalised_cost_inr(
        self,
        in_vehicle_time_min: float,
        waiting_time_min: float,
        access_time_min: float,
        fare_inr: float,
        occupancy_ratio: float,
        has_realtime_info: bool = False,
        segment: PassengerSegment = PassengerSegment.OCCASIONAL_USER,
    ) -> float:
        """Converts the utility to a generalised cost in INR using WTP weights.

        Generalised Cost = Fare + VoTT × TT + VoWT × WT + VoAT × AT + Crowding_Penalty
        where VoTT (Value of Travel Time) = WTP per minute.

        This is the form used in PUMO's assignment scoring — replacing
        the old arbitrary scalar weights with empirically calibrated WTP values.
        """
        coeff = self.base_coefficients  # Use population mean for consistent scoring
        scales = SEGMENT_COEFFICIENT_SCALES[segment]

        wtp_tt = abs(coeff.beta_travel_time * scales["travel_time"] / (coeff.beta_fare * scales["fare"]))
        wtp_wt = abs(coeff.beta_waiting_time * scales["waiting_time"] / (coeff.beta_fare * scales["fare"]))
        wtp_at = abs(coeff.beta_access_time * scales["access_time"] / (coeff.beta_fare * scales["fare"]))
        wtp_rt = abs(coeff.beta_realtime_info * scales["realtime_info"] / (coeff.beta_fare * scales["fare"]))
        crowding_cost = abs(coeff.beta_crowding * scales["crowding"] / (coeff.beta_fare * scales["fare"]))

        crowding = crowding_discomfort(occupancy_ratio)
        rt_saving = wtp_rt if has_realtime_info else 0.0

        gc = (
            fare_inr
            + wtp_tt * in_vehicle_time_min
            + wtp_wt * waiting_time_min
            + wtp_at * access_time_min
            + crowding_cost * crowding
            - rt_saving  # Real-time information reduces perceived cost
        )
        return max(0.0, gc)

    def wtp_summary(self, segment: PassengerSegment = PassengerSegment.OCCASIONAL_USER) -> Dict[str, float]:
        """Returns WTP estimates in INR/min for the given segment."""
        bc = self.base_coefficients
        scales = SEGMENT_COEFFICIENT_SCALES[segment]
        fare_scale = scales["fare"]
        effective_fare_beta = bc.beta_fare * fare_scale
        return {
            "segment": segment.value,
            "wtp_in_vehicle_time_per_min_inr": round(
                abs(bc.beta_travel_time * scales["travel_time"] / effective_fare_beta), 3),
            "wtp_waiting_time_per_min_inr": round(
                abs(bc.beta_waiting_time * scales["waiting_time"] / effective_fare_beta), 3),
            "wtp_access_time_per_min_inr": round(
                abs(bc.beta_access_time * scales["access_time"] / effective_fare_beta), 3),
            "wtp_realtime_info_per_trip_inr": round(
                abs(bc.beta_realtime_info * scales["realtime_info"] / effective_fare_beta), 3),
        }
