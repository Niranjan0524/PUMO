"""W: seeded weighted-sum sweeps followed by a shared Pareto filter."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from pumo.model import Query, RouteResult, validate_result
from pumo.objectives import OBJECTIVES
from pumo.pareto import pareto_filter
from .base import Router
from .weighted_astar import WeightedAStarRouter


class WeightSweepRouter(Router):
    name = "W"

    def __init__(self, network=None, *, scales=None, sweep_size=50, seed=20261016,
                 profiles=None, label_cap=2_000_000, max_speed_cap_mps=40.0):
        super().__init__(network, label_cap=label_cap)
        self.scales = dict(scales or {key: 1.0 for key in OBJECTIVES})
        self.sweep_size = int(sweep_size)
        self.seed = int(seed)
        self.max_speed_cap_mps = max_speed_cap_mps
        if self.sweep_size <= 0:
            raise ValueError("sweep_size must be positive")
        if profiles is None:
            path = Path(__file__).resolve().parents[3] / "config" / "profiles.csv"
            if path.exists():
                frame = pd.read_csv(path).set_index("profile_id")
                column_map = dict(zip(OBJECTIVES, frame.columns))
                profiles = [{key: float(frame.loc[index, column_map[key]])
                             for key in OBJECTIVES} for index in frame.index]
            else:
                profiles = []
        self.profiles = list(profiles)

    def weight_vectors(self, objectives):
        """Unit vectors, configured profiles, then seeded Dirichlet samples."""
        names = tuple(objectives)
        if not names:
            raise ValueError("at least one objective is required")
        vectors = [{key: float(key == name) for key in names} for name in names]
        for profile in self.profiles:
            restricted = {key: float(profile.get(key, 0.0)) for key in names}
            if any(not np.isfinite(value) or value < 0 for value in restricted.values()):
                raise ValueError("profile weights must be finite and non-negative")
            total = sum(restricted.values())
            if total > 0:
                vectors.append({key: value / total for key, value in restricted.items()})

        # Mandatory unit/profile vectors are retained even if the configured
        # sweep size is smaller than their count. Otherwise fill exactly to N.
        remaining = max(0, self.sweep_size - len(vectors))
        if remaining:
            rng = np.random.default_rng(self.seed)
            vectors.extend(dict(zip(names, map(float, sample)))
                           for sample in rng.dirichlet(np.ones(len(names)), size=remaining))
        return vectors

    def route(self, q: Query) -> RouteResult:
        if self.network is None:
            raise ValueError("router has no PreparedNetwork")
        vectors = self.weight_vectors(q.objectives)
        started = time.perf_counter()
        journeys = []
        run_stats = []
        router = WeightedAStarRouter(self.network, scales=self.scales,
                                      label_cap=self.label_cap,
                                      max_speed_cap_mps=self.max_speed_cap_mps)
        for weights in vectors:
            weighted_query = Query(q.origin, q.destination, q.departure_time,
                                   q.objectives, weights, q.disruptions,
                                   q.start_state, q.max_transfers, q.timeout_s)
            result = router.route(weighted_query)
            run_stats.append(result.stats)
            journeys.extend(result.journeys)

        by_path = {}
        for journey in journeys:
            segment_ids = tuple(segment for leg in journey.legs for segment in leg.segment_ids)
            by_path.setdefault(segment_ids, journey)
        selected = tuple(q.objectives)
        result_journeys = pareto_filter(
            by_path.values(), lambda journey: tuple(journey.vector.get(key, 0.0) for key in selected))
        total_runtime = time.perf_counter() - started
        per_weight_runtime = [float(stats.get("runtime_s", 0.0)) for stats in run_stats]
        result = RouteResult(result_journeys, {
            "runtime_s": total_runtime,
            "total_runtime_s": total_runtime,
            "sweep_size_requested": self.sweep_size,
            "sweep_size": len(vectors),
            "weight_vectors_run": len(vectors),
            "weight_vectors": vectors,
            "per_weight_runtime_s": per_weight_runtime,
            "distinct_journeys": len(by_path),
            "mean_runtime_per_weight_s": sum(per_weight_runtime) / max(1, len(per_weight_runtime)),
            "labels_generated": sum(stats.get("labels_generated", 0) for stats in run_stats),
            "labels_expanded": sum(stats.get("labels_expanded", 0) for stats in run_stats),
            "timed_out": any(stats.get("timed_out", False) for stats in run_stats),
            "label_cap_hit": any(stats.get("label_cap_hit", False) for stats in run_stats),
            "exact": all(stats.get("exact", False) for stats in run_stats),
            "supported_front_only": True,
            "limitation": "weighted sums can miss unsupported Pareto points",
        })
        validate_result(result, objectives=selected, pareto=True)
        return result
