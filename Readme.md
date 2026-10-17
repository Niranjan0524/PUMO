# Personal Urban Mobility Optimizer (PUMO)

PUMO is an urban mobility optimization research platform designed for megacities like Bengaluru. It features modular routing subsystems addressing multimodal transit, safety-aware navigation, and dynamic on-demand paratransit.

---

## Sub-Topics & Modules

### 1. [Paratransit Routing](./paratransit_routing)
Dynamic Dial-a-Ride Problem (D-DARP) algorithms for flexible shared auto-rickshaws:
- **Graph & Travel-Time Routing (A1)**: Directed road graph with memoized Dijkstra shortest path.
- **Baseline Dispatch Heuristics (A2, A3)**: Nearest Available Vehicle (unshared) and Nearest Feasible Vehicle (sequential append).
- **Shared-Route Optimization (A4, A5)**: Greedy Insertion Heuristic and Regret-k Insertion Heuristic with capacity, time-window, and existing-passenger preservation constraints.
- **Multi-Objective Optimization (A6)**: Pareto-based feasible insertion balancing passenger journey time and shared-auto fare (Time-First, Fare-First, and Balanced).
- **Exact Solver Benchmark (A7)**: Combinatorial branch-and-bound solver establishing proven global optimum and optimality gap metrics.
- **Evaluation**: Empirical benchmarks across Low, Medium, and High demand scenarios, with documented fare models and performance figures.

### 2. [Multi-Objective Transit Routing](./Multi-Objective%20Transit%20Routing)
Public transit routing across bus (BMTC) and metro networks:
- Timetable-based journey planning.
- Multi-objective Pareto frontier generation (travel time, transfers, walking distance).

### 3. [Safety-Aware Routing](./safety_routing)
Road-level safety integration:
- Streetlight density matching and flood-prone area spatial analysis.
- Multi-attribute safety-aware cost functions.