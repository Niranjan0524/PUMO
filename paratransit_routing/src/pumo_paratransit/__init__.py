"""PUMO Paratransit Routing Module.

Flexible shared-auto and dial-a-ride routing algorithms, simulation,
and multi-objective evaluation.
"""

from .models import (
    StopType,
    Stop,
    PassengerRequest,
    RequestStatus,
    Vehicle,
    AssignmentDecision,
)
from .network import RoadGraph, create_bengaluru_synthetic_network, create_small_test_graph
from .dijkstra import DijkstraRouter
from .fare import ParatransitFareCalculator
from .constraints import RouteConstraintChecker
from .baselines import NearestAvailableAssigner, NearestFeasibleAssigner
from .insertion import GreedyInsertionAssigner
from .regret import RegretInsertionAssigner
from .pareto import ParetoMultiObjectiveAssigner
from .exact_solver import ExactDARPSolver
from .simulator import ParatransitSimulator
from .metrics import SimulationMetrics, compute_simulation_metrics

__all__ = [
    "StopType",
    "Stop",
    "PassengerRequest",
    "RequestStatus",
    "Vehicle",
    "AssignmentDecision",
    "RoadGraph",
    "create_bengaluru_synthetic_network",
    "create_small_test_graph",
    "DijkstraRouter",
    "ParatransitFareCalculator",
    "RouteConstraintChecker",
    "NearestAvailableAssigner",
    "NearestFeasibleAssigner",
    "GreedyInsertionAssigner",
    "RegretInsertionAssigner",
    "ParetoMultiObjectiveAssigner",
    "ExactDARPSolver",
    "ParatransitSimulator",
    "SimulationMetrics",
    "compute_simulation_metrics",
]
