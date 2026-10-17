# Mathematical Problem Formulation: Dynamic Dial-a-Ride Problem (D-DARP) for Shared Autos

## 1. Problem Overview
The Paratransit Routing module in PUMO models on-demand, flexible shared-auto services (such as shared auto-rickshaws in Bengaluru).
The problem belongs to the **Dynamic Dial-a-Ride Problem (D-DARP)** family. Passenger requests arrive dynamically over time. Autos can deviate from their direct routes to pick up and drop off multiple passengers with overlapping itineraries, subject to vehicle capacity, pickup waiting-time limits, and in-vehicle detour constraints.

---

## 2. Road Network and Travel Times
Let $G = (V_N, E)$ be a directed road graph:
- $V_N$: Set of network vertices (intersections, transit hubs, activity centers).
- $E \subseteq V_N \times V_N$: Directed road segments with distance $d(u, v) > 0$ (km) and travel time $t(u, v) > 0$ (seconds).
- For any pair of nodes $u, v \in V_N$, the minimum travel time $T^*(u, v)$ and shortest path distance $D^*(u, v)$ are computed via Dijkstra's algorithm ($A_1$) and memoized.

---

## 3. Request Model
A passenger request $r \in \mathcal{R}$ is defined by the tuple:
$$r = (id_r, o_r, d_r, t_r^{\text{req}}, q_r, W_r^{\max}, \alpha_r^{\max})$$
where:
- $o_r \in V_N$: Origin node.
- $d_r \in V_N$: Destination node.
- $t_r^{\text{req}} \ge 0$: Request submission timestamp.
- $q_r \in \mathbb{N}_{\ge 1}$: Number of passengers in the party.
- $W_r^{\max} > 0$: Maximum acceptable waiting time (e.g., 600–900 seconds).
- $\alpha_r^{\max} \ge 1.0$: Maximum acceptable detour ratio (e.g., 1.5 – 1.8x of direct travel time).

### Derived Service Deadlines:
1. **Pickup Deadline**:
   $$T_r^{\text{pickup, max}} = t_r^{\text{req}} + W_r^{\max}$$
2. **Direct In-Vehicle Travel Time**:
   $$T_r^{\text{direct}} = T^*(o_r, d_r)$$
3. **Drop-Off Deadline**:
   $$T_r^{\text{dropoff, max}} = T_r^{\text{pickup}} + \alpha_r^{\max} \cdot T_r^{\text{direct}}$$

---

## 4. Vehicle Model
Let $\mathcal{K}$ be the fleet of shared autos. Each vehicle $k \in \mathcal{K}$ at time $t$ is characterized by:
- $C_k \in \mathbb{N}$: Vehicle passenger capacity (e.g., 3–4 seats for auto-rickshaws, 6 for e-shuttles).
- $loc_k(t) \in V_N$: Current location or upcoming node.
- $L_k(t) \in [0, C_k]$: Current occupied seats.
- $\Pi_k = (s_1, s_2, \dots, s_m)$: Current planned stop sequence.
- Each stop $s_i$ contains:
  - $node(s_i) \in V_N$: Location of the stop.
  - $req(s_i) \in \mathcal{R}$: Associated passenger request.
  - $type(s_i) \in \{\text{PICKUP}, \text{DROPOFF}\}$.
  - $\Delta q(s_i) \in \{-q_r, +q_r\}$: Passenger load change.
  - $t^{\text{arr}}(s_i), t^{\text{dep}}(s_i)$: Scheduled arrival and departure times.

---

## 5. Feasibility Constraints

When inserting a new request $r$ into vehicle $k$'s route at pickup position $i$ and drop-off position $j$ ($0 \le i \le j \le m$):

1. **Precedence Constraint**:
   $$i \le j$$
   The passenger must be picked up before being dropped off.

2. **Capacity Constraint**:
   For every segment $p \in \{0, \dots, m+2\}$ along the updated route:
   $$Load_k(p) = L_k(t) + \sum_{\ell=1}^p \Delta q(s'_\ell) \le C_k$$
   Vehicle capacity is never exceeded.

3. **New Passenger Time-Window Constraints**:
   - $t^{\text{arr}}(s'_i) \ge t_r^{\text{req}}$ (cannot pick up before request is placed).
   - $t^{\text{arr}}(s'_i) \le T_r^{\text{pickup, max}}$ (waiting time constraint).
   - $t^{\text{arr}}(s'_j) - t^{\text{dep}}(s'_i) \le \alpha_r^{\max} \cdot T_r^{\text{direct}}$ (ride time / detour constraint).

4. **Existing Passenger Preservation Constraints**:
   For all existing committed requests $r' \in \Pi_k$:
   - If $r'$ has not been picked up: $t^{\text{arr, new}}(\text{pickup}_{r'}) \le T_{r'}^{\text{pickup, max}}$.
   - $t^{\text{arr, new}}(\text{dropoff}_{r'}) - t_{\text{actual/planned}}^{\text{pickup}}(r') \le \alpha_{r'}^{\max} \cdot T_{r'}^{\text{direct}}$.
   No existing passenger's promised maximum wait or detour limit is violated.

---

## 6. Objectives and Trade-Offs

1. **Passenger Journey Time**:
   $$J_r = (T_r^{\text{pickup}} - t_r^{\text{req}}) + (T_r^{\text{dropoff}} - T_r^{\text{pickup}}) = T_r^{\text{dropoff}} - t_r^{\text{req}}$$
2. **Passenger Fare**:
   $$F_r = \text{Fare}(r, \text{detour}, \text{shared\_passengers})$$
3. **Vehicle Distance / Empty Running**:
   Minimizing total vehicle distance traveled while maximizing request service rate.

In Phase 5, the multi-objective approach evaluates $(J_r, F_r)$ directly to retain Pareto-optimal choices (Time-First, Fare-First, and Balanced).
