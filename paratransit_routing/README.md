# PUMO Paratransit & Shared-Auto Routing

This module designs, implements, and benchmarks dynamic on-demand routing algorithms for flexible shared-auto and paratransit services within the **Personal Urban Mobility Optimizer (PUMO)**.

The paratransit routing problem is formulated as a **Dynamic Dial-a-Ride Problem (D-DARP)**: requests arrive dynamically over time, passengers share vehicles with overlapping trips, and routes are continually adapted subject to vehicle capacity, pickup waiting-time deadlines, maximum acceptable detours, and existing passenger commitments.

---

## Algorithms Implemented

| ID | Algorithm | Type | Description |
|---|---|---|---|
| **A1** | **Dijkstra Travel-Time Router** | Core Graph | Directed shortest path with all-pairs memoization for fast travel-time and distance queries. |
| **A2** | **Nearest Available Vehicle (NAV)** | Baseline | Dispatches the nearest idle vehicle (solo ride-hail, no sharing). Rejects if no vehicle is free. |
| **A3** | **Nearest Feasible Vehicle (NFV)** | Baseline | Appends the request sequentially to vehicles with spare capacity without route interleaving. |
| **A4** | **Greedy Insertion Heuristic (GIH)** | Main Routing | Combinatorially evaluates all feasible $(i, j)$ pickup/drop-off insertion positions across all vehicles. |
| **A5** | **Regret-2 Insertion Heuristic (RIH)** | Strong Baseline | Prioritizes requests with high regret ($c_2 - c_1$) to protect difficult-to-serve requests. |
| **A6** | **Pareto-Based Feasible Insertion (PFI)** | Multi-Objective | Evaluates candidate insertions on $(T_{\text{journey}}, \text{Fare})$ space, pruning dominated options into Time-First, Fare-First, and Balanced choices. |
| **A7** | **Exact Optimization Solver** | Benchmark | Branch-and-bound combinatorial solver finding proven global optimal solutions on benchmark instances to evaluate heuristic optimality gaps. |

---

## Directory Structure

```
paratransit_routing/
├── README.md                          # Module overview and empirical results
├── FARE_MODEL.md                      # Fare pricing equations and behavioral assumptions
├── docs/
│   └── PROBLEM_FORMULATION.md         # Mathematical D-DARP formulation & constraints
├── src/
│   └── pumo_paratransit/
│       ├── __init__.py                # Package root exports
│       ├── models.py                  # Stop, PassengerRequest, Vehicle, AssignmentDecision
│       ├── network.py                 # RoadGraph, Bengaluru 15-node network, test graphs
│       ├── dijkstra.py                # A1: Dijkstra shortest path router with caching
│       ├── fare.py                    # Bengaluru-based paratransit fare calculation engine
│       ├── constraints.py             # Precedence, capacity, time windows, and preservation checker
│       ├── baselines.py               # A2 (Nearest Available) & A3 (Nearest Feasible)
│       ├── insertion.py               # A4 (Greedy Insertion Heuristic)
│       ├── regret.py                  # A5 (Regret-k Insertion Heuristic)
│       ├── pareto.py                  # A6 (Pareto Multi-Objective Insertion)
│       ├── exact_solver.py            # A7 (Exact DARP Combinatorial Solver)
│       ├── simulator.py               # Dynamic discrete-event simulation engine
│       └── metrics.py                 # Metrics aggregator and evaluator
├── tests/                             # Full unit test suite (23 tests, 100% passing)
│   ├── test_network_dijkstra.py       # A1 tests & graph verification
│   ├── test_fare.py                   # Fare calculation & discount verification
│   ├── test_constraints.py            # Hard constraint violations & preservation
│   ├── test_baselines.py              # A2 & A3 tests
│   ├── test_insertion_heuristics.py   # A4 & A5 tests
│   ├── test_pareto.py                 # A6 Pareto front & dominance tests
│   ├── test_exact_solver.py           # A7 Exact solver tests
│   └── test_simulation.py             # Dynamic end-to-end simulation tests
├── scripts/
│   ├── generate_scenarios.py          # Generates reproducible Low, Med, High, and Exact datasets
│   ├── 01_run_unit_tests.py           # Runs test suite
│   ├── 02_run_benchmark.py            # Runs benchmark across Low, Medium, High demand tiers
│   ├── 03_generate_plots.py           # Generates comparison plots (PNG)
│   └── 04_exact_gap_evaluation.py     # Evaluates heuristics against exact global optimum (A7)
├── data/
│   ├── synthetic_network.json         # Bengaluru 15-hub network definition
│   ├── scenario_low.json              # 20 requests, 4 vehicles
│   ├── scenario_medium.json           # 60 requests, 8 vehicles
│   ├── scenario_high.json             # 150 requests, 12 vehicles
│   └── scenario_exact.json            # 6 requests, 2 vehicles validation instance
└── results/
    ├── benchmark_summary.csv          # Formatted comparison table
    ├── benchmark_results.json         # Complete raw metrics
    ├── exact_optimality_gap.csv       # Exact benchmark comparison table
    ├── exact_optimality_gap.json      # Exact benchmark raw data
    └── figures/
        ├── service_rate_by_demand.png
        ├── waiting_and_journey_time.png
        └── optimality_gap.png
```

---

## Empirical Benchmark Results

### Demand Tier Comparison (Low, Medium, High)

| Scenario | Algorithm | Service Rate (%) | Avg Wait (min) | Avg Ride (min) | Avg Fare (₹) | Avg Detour | Fleet Dist (km) | Runtime (ms/req) |
|---|---|---|---|---|---|---|---|---|
| **Low** (20 req, 4 veh) | A2: Nearest Available | 40.0% | 7.60 | 18.99 | ₹136.5 | 1.04 | 101.6 | 0.03 |
| | A3: Nearest Feasible | 40.0% | 7.98 | 20.82 | ₹170.6 | 1.03 | 105.4 | 0.08 |
| | **A4: Greedy Insertion** | **65.0%** | **5.14** | 24.31 | ₹149.7 | 1.15 | 113.8 | 0.35 |
| | **A5: Regret Insertion** | **65.0%** | **5.14** | 24.31 | ₹149.7 | 1.15 | 113.8 | 0.56 |
| | **A6: Pareto (Balanced)**| **65.0%** | **5.11** | 24.39 | ₹149.5 | 1.16 | 113.8 | 0.43 |
| **Medium** (60 req, 8 veh) | A2: Nearest Available | 31.7% | 5.71 | 19.74 | ₹139.1 | 1.03 | 228.9 | 0.02 |
| | A3: Nearest Feasible | 35.0% | 7.13 | 21.70 | ₹172.4 | 1.03 | 270.8 | 0.18 |
| | **A4: Greedy Insertion** | **61.7%** | **4.36** | 30.72 | ₹162.0 | 1.24 | 296.2 | 1.75 |
| | **A5: Regret Insertion** | **61.7%** | **4.36** | 30.72 | ₹162.0 | 1.24 | 296.2 | 1.33 |
| | **A6: Pareto (Balanced)**| **55.0%** | **4.89** | 29.73 | ₹166.4 | 1.27 | 288.2 | 1.19 |
| **High** (150 req, 12 veh) | A2: Nearest Available | 18.7% | 4.99 | 24.19 | ₹177.9 | 1.03 | 400.3 | 0.02 |
| | A3: Nearest Feasible | 20.7% | 8.10 | 26.83 | ₹222.6 | 1.02 | 464.3 | 0.29 |
| | **A4: Greedy Insertion** | **44.0%** | **4.84** | 31.62 | ₹194.4 | 1.23 | 518.0 | 2.42 |
| | **A5: Regret Insertion** | **44.0%** | **4.84** | 31.62 | ₹194.4 | 1.23 | 518.0 | 2.16 |
| | **A6: Pareto (Balanced)**| **44.0%** | **5.14** | 32.76 | ₹198.3 | 1.25 | 504.5 | 2.59 |

---

### Exact Optimality Gap Benchmark (vs A7 Global Optimum)

Evaluated on the exact validation instance (6 dynamic requests, 2 shared autos, small test network):

| Algorithm | Served Requests | Service Rate (%) | Total Journey (s) | Total Vehicle Dist (km) | Combined Cost | Optimality Gap (%) |
|---|---|---|---|---|---|---|
| **A7: Exact Solver** | **6 / 6** | **100.0%** | **5,923.3 s** | **30.0 km** | **5,938.30** | **0.00% (Global Optimum)** |
| **A4: Greedy Insertion** | **6 / 6** | **100.0%** | 6,328.3 s | 30.0 km | 6,343.30 | **+6.82%** |
| **A5: Regret Insertion** | **6 / 6** | **100.0%** | 6,328.3 s | 30.0 km | 6,343.30 | **+6.82%** |
| A6: Pareto Multi-Objective | 5 / 6 | 83.3% | 4,861.8 s | 33.0 km | 24,878.30 | +318.95% |
| A3: Nearest Feasible | 4 / 6 | 66.7% | 4,604.4 s | 33.0 km | 44,620.90 | +651.41% |
| A2: Nearest Available | 3 / 6 | 50.0% | 3,450.0 s | 32.0 km | 63,466.00 | +968.76% |

#### Key Takeaways:
1. **Sharing dramatically boosts capacity**: As demand rises (High tier), unshared solo dispatch (A2) collapses to 18.7% service rate, while insertion heuristics (A4, A5) maintain a 44.0% service rate (>2.3x higher).
2. **Minimal Detour Overhead**: Greedy and Regret insertion achieve higher throughput with an average detour ratio of only $1.15$ to $1.27$ (passengers spend only 15–27% extra time in vehicle while enjoying shared-fare discounts).
3. **Proven Near-Optimality**: On the exact benchmark instance, Greedy Insertion (A4) and Regret Insertion (A5) achieve an optimality gap of only **6.82%** relative to the global theoretical optimum (A7) while computing decisions in less than 3 milliseconds per request.

---

## Reproducing the Results

Run all unit tests:
```powershell
python paratransit_routing/scripts/01_run_unit_tests.py
```

Run the multi-tier benchmark:
```powershell
python paratransit_routing/scripts/02_run_benchmark.py
```

Run the exact solver optimality gap evaluation:
```powershell
python paratransit_routing/scripts/04_exact_gap_evaluation.py
```

Generate visual plots:
```powershell
python paratransit_routing/scripts/03_generate_plots.py
```
