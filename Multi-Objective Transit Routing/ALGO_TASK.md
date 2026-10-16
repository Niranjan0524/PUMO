# ALGO_TASK.md — Build & Benchmark Multi-Objective Multimodal Routing Algorithms (Bengaluru)

> Give this file to the coding agent (Codex / Claude / Cursor) together with `CONTEXT.md`.
> `CONTEXT.md` = what the project is and general data rules. **This file = the specific task:** build a common data representation from the downloaded files, implement several routing algorithms ourselves, benchmark them fairly, and conclude which is best.
> If the two files disagree on a detail about this task, this file wins.
>
> **Final algorithm set (decided by the human): B0 (weighted-sum A\*), W (weight-sweep), A2 (McRAPTOR-style), A1 (exact label-setting, Core region only).** ML-TZSA and NSGA-II are covered in the literature comparison only and are **not** to be implemented.

---

## 0. Working agreement (read first)

1. **Inspect before assuming.** The human's data sits in `Data/` (see Section 2). Contents of subfolders are unknown to you until you open them. Run an inventory first and print what you found.
2. **Never fabricate data.** If something is missing, use the fallbacks in Section 4.9 (all clearly labeled `PLACEHOLDER` / `IMPUTED`) and keep going. Do not stop to ask unless you are truly blocked (e.g. the OSM file or every transit feed is unreadable).
3. **Work in phases** (Section 11). After each phase: run tests, commit, and print a short status (what was built, what was skipped, any warnings). Don't build everything in one shot.
4. **Fairness is the whole point.** Every algorithm must read the *same* prepared data through the *same* interface, with the *same* objective definitions and normalization, at a similar level of optimization (Section 6.1). Don't quietly optimize one algorithm more than the others.
5. **Correctness before speed.** Write a plain, readable reference implementation first. Only add optimizations (numpy arrays, numba) if they don't change results, and re-run the tests afterwards.
6. **Decide the comparison rule before you see results** (Section 8.4). Write it to `docs/BENCHMARK_PROTOCOL.md` and commit it *before* the first benchmark run.
7. **Be honest in outputs.** The conclusion is "best on this Bengaluru dataset for these metrics", never "best in general". State every limitation.
8. Log non-obvious choices (thresholds, dropped records, imputation) in `docs/DECISIONS.md`.

---

## 1. Goal

### 1.1 Project in one paragraph
PUMO (Personalized Urban Mobility Optimizer) is a final-year B.Tech project: a personalized multimodal route planner for **Bengaluru** (walking + BMTC bus + Namma Metro). Every journey is scored on six objectives that are all **minimized**:

```
J = [T, C, X, W, E, S]
T = travel time (arrival − departure)      C = cost
X = number of transfers                    W = walking burden
E = estimated emissions                    S = safety-risk exposure
```

### 1.2 This task (one team member's category)
Compare different **multi-objective multimodal routing algorithms** that appear in the literature the team reviewed, implement them ourselves on a **common dataset**, benchmark them, and conclude which one is best for PUMO's purpose.

Algorithms to implement (details in Section 5) — **four**, chosen because they are lightweight and fit one shared dataset:

| ID | Algorithm | Returns | Role in the comparison |
| --- | --- | --- | --- |
| **B0** | Weighted-sum A\* (time-dependent) | One journey per preference profile | Simplest baseline; shows what a single-score method misses (alternatives, trade-off diversity) |
| **W** | Weight-sweep: run B0 with many weight vectors, then Pareto-filter | Approximate Pareto set | Cheap approximation; limited to "supported" Pareto points (the convex part of the trade-off), so it can miss balanced routes — that limit is part of the result |
| **A2** | McRAPTOR-style round-based multi-criteria search (with walking transfers) | Pareto set (exact within its model) | Well-known multi-criteria transit method, fast at query time |
| **A1** | Exact multi-objective label-setting search (Pareto Dijkstra / NAMOA\*-style) | Full Pareto set | **Exact reference** for scoring the others. Run **only on the small Core region**, with a label cap, because it is the slow one |

**Not implemented (literature comparison only):** the ML-TZSA transfer-zone method and NSGA-II (used by MODyPer). They appear in the written comparison table (what they do, what they need, why they were not built) and as future work.

### 1.3 Papers behind the algorithms (for context and naming)
- **RAPTOR / McRAPTOR (Delling et al.)** — round-based public-transit routing on timetables; round *k* = best arrival using *k* trips; multi-criteria extension keeps Pareto bags.
- **MaaS ML-TZSA (Bayliss et al., 2025)** — multimodal MaaS routing using *transfer zones*, sampled zone sequences, precomputed static contributions, Pareto journey profiles, then exact transfer-point optimization; a neural net only predicts downstream travel time. **Literature comparison only — not implemented here** (it needs a trained predictor and is the hardest to build).
- **OPMOS (Gold et al., 2024)** — exact multi-objective search is bottlenecked by dominance checks; label counts explode as objectives grow. Motivates scaling tests with 2 → 6 objectives.
- **MODyPer (Rollo & Po, 2025)** — Pareto candidates + user weights for walking/cycling; uses NSGA-II. Its "generate candidates, then Pareto-filter, then apply user weights" pattern is the inspiration for **W**; NSGA-II itself is **not implemented**.
- **Price-optimal routing (Euler et al.)** — path-dependent fares; **not** implemented here (we use additive cost; see 4.9).
- **Learn-n-Route** and the **route recommendation survey** — background only.

### 1.4 Deliverables
1. Python package implementing the data preparation, the algorithms, and the benchmark runner.
2. Passing test suite (Section 9).
3. Benchmark results as CSV + figures.
4. `reports/ALGO_COMPARISON.md`: methods, tables, conclusion following the pre-set rule, limitations.
5. `reports/SLIDE_SUMMARY.md`: one-page summary (what each algorithm is, what was measured, which won and why) that the human can turn into slides.

---

## 2. The data the human has

The human's folder (as shown in their editor):

```
Data/
├── bmrcl/                        # folder — probably Namma Metro GTFS (inspect!)
├── bmtc/                         # folder — probably BMTC bus GTFS (inspect!)
├── Accident data.csv
├── floodZone data.kml
├── southern-zone-261013.osm.pbf  # Geofabrik "Southern Zone" OSM extract (name suggests 13 Oct 2026)
└── streetlight data.csv
```

Treat `Data/` as **read-only**. Copy/organize into `data/raw_organized/` if you want a tidier layout, but never edit the originals. File names contain spaces — handle paths robustly.

### 2.1 What each item is and how it is used

| Item | Expected content | Used for |
| --- | --- | --- |
| `southern-zone-*.osm.pbf` | Large OSM extract for southern India (hundreds of MB) | Walking/road graph, traffic signals/crossings, optional OSM streetlamps, transit-entrance geometry. **Must be clipped to Bengaluru + buffer first.** |
| `bmtc/` | Probably GTFS (`stops.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`, `calendar.txt`, maybe `shapes.txt`, `fare_*.txt`) as loose files or a `.zip`; there may be more than one feed | Bus timetable network |
| `bmrcl/` | Probably metro GTFS, same file set. Intermediate stop times may be *approximated* by whoever generated the feed | Metro timetable network |
| `streetlight data.csv` | Streetlight records; may have lat/lon, or only ward/zone counts | Lighting component of the safety score |
| `floodZone data.kml` | KML points/polygons of flood-prone locations | Flood component of the safety score |
| `Accident data.csv` | Accident records; **may be aggregate** (city/year) rather than point-level | Accident component *only if* spatial (lat/lon or mappable location); else context only |

### 2.2 Step 0 — inventory (do this first)
Write `scripts/00_inventory.py` that scans `Data/` recursively and records for each file: size, extension, sha256, and a content summary:
- CSV/XLSX: row count, columns, dtypes, null %, 5 sample rows, whether latitude/longitude-like columns exist (names such as `lat`, `latitude`, `lon`, `lng`, `longitude`, `x`, `y`, or a single combined `location` column).
- KML: layers, feature count per layer, geometry types, bounding box, attribute fields.
- GTFS (folder or zip): which required files exist, agency name(s), #stops/#routes/#trips/#stop_times, `calendar` date range, whether fares exist.
- PBF: file size and, if readable, the bounding box.

Save `data/INVENTORY.md` + `data/inventory.json` and print a summary table: **found / unreadable / unclassified**. Decide BUS vs METRO from agency/route names (metro: "Purple Line", "Green Line", "Namma Metro", BMRCL; bus: numbered routes), not from folder names alone. If you find two bus feeds, keep both and choose a primary using the calendar rule in Section 4.4.

---

## 3. Design overview

```
Data/ (raw)  ──►  preprocessing  ──►  PreparedNetwork (one common in-memory + on-disk format)
                                              │
              ┌───────────────┬───────────────┬───────────────┐
              ▼               ▼               ▼               ▼
        A1 label-setting   A2 McRAPTOR   B0 weighted A*   W weight-sweep
        (Core only)                                       (= many B0 runs)
              └───────────────┴───────────────┴───────────────┘
                                     same Query / Result interface
                                              │
                         benchmark runner  ──►  metrics  ──►  report
```

Everything downstream of preprocessing reads **only** the `PreparedNetwork`. No algorithm reads raw files.

Suggested package layout (create under the repo; adapt names if the repo already has a convention):

```
src/pumo/
  data/            # loaders: osm.py, gtfs.py, evidence.py, safety.py, build_network.py
  network.py       # PreparedNetwork dataclasses + (de)serialization
  objectives.py    # objective vector definitions, normalization
  pareto.py        # dominance, Pareto filter, epsilon-dominance, hypervolume
  query.py         # Query, Journey, Leg, RouteResult, Disruption, JourneyState
  algos/
    base.py        # abstract Router
    label_setting.py   # A1 (Core region only)
    mcraptor.py        # A2
    weighted_astar.py  # B0
    weight_sweep.py    # W (calls B0 repeatedly)
  bench/
    scenarios.py   # OD-pair generation
    runner.py      # run algorithms, time/memory, timeouts
    metrics.py     # recall, hypervolume, regret, ...
    report.py      # tables + figures
tests/
config/            # benchmark.yaml, thresholds.yaml, safety_weights.yaml, emission_factors.csv, profiles.csv
docs/              # BENCHMARK_PROTOCOL.md, DECISIONS.md, SCHEMA.md
reports/
```

---

## 4. Data representation and preprocessing

### 4.1 Global conventions
- Store geometry in **EPSG:4326**; do all length/distance/buffer/nearest-neighbour work in **EPSG:32643** (UTM zone 43N, suitable for Bengaluru).
- **Time** is integer **seconds after midnight of the service date**; GTFS times ≥ 24:00:00 are valid (e.g. `25:10:00` = 90600) — never wrap them.
- Use **integer IDs** internally (re-map OSM node IDs, stop IDs, trip IDs to 0..N−1) and keep the original IDs in lookup tables.
- Use numpy arrays / CSR-style adjacency for anything in a hot loop. Don't run routing loops through `networkx`; it's fine for offline utilities and tests.
- Cache prepared outputs to `data/processed/` (Parquet / GeoPackage / `.npz`) so benchmarks don't re-run preprocessing.

### 4.2 Build the walking graph from OSM
1. Clip the PBF to **Bengaluru + buffer**. Approximate bounding box *to be verified against the real boundary*: lon ≈ 77.35–77.85, lat ≈ 12.75–13.25. Prefer a polygon clip (city boundary from OSM admin tags, buffered ~3 km, configurable). Save the clipped file under `data/interim/`.
2. Extract pedestrian-traversable ways: `footway`, `pedestrian`, `path`, `steps`, `living_street`, `residential`, `service`, `unclassified`, `tertiary`, `secondary`, `primary`, and `crossing` ways where mapped. Exclude motorways/trunk and ways with `foot=no` / `access=no` (for walking).
3. Build **directed edges**: two-way → two edges; `oneway` doesn't restrict pedestrians by default (for walking, treat roads as two-way unless `oneway:foot` or access tags say otherwise; document the choice).
4. Edge attributes: `edge_id, from_node, to_node, osm_way_id, length_m` (computed in EPSG:32643), `road_class`, `geometry`.
5. Walking time = `length_m / walk_speed` (default 1.33 m/s ≈ 4.8 km/h, configurable). Mark it MODELED.
6. Keep the largest connected component for routing; log the share kept and the isolated fragments dropped.
7. Pull `highway=traffic_signals`, `highway=crossing`, and (as lighting fallback) `highway=street_lamp` nodes for the safety layer.

### 4.3 The transit timetable (GTFS)
For each feed (bus, metro):
1. Load `stops`, `routes`, `trips`, `stop_times`, `calendar`, `calendar_dates` (if present), and fares if present.
2. **Choose a service date** (config key `service_date`; default = first weekday that lies inside the date range covered by *both* feeds; if the ranges don't overlap, pick a weekday inside each feed's range and run the two feeds on their own date mapped to the same time-of-day — document this in `DECISIONS.md`). Use only trips whose `service_id` is active that day.
3. **Validate**: stop coordinates inside the Bengaluru region; `arrival <= departure`; positive time progression along each trip; drop/quarantine failures into `data/quarantine/` with a reason.
4. Build:
   - `stops`: `stop_idx, stop_id, agency, mode (BUS|METRO), name, lat, lon, node_idx` (snapped walking-graph node), `snap_dist_m`
   - `routes`/`trips`: trips grouped into **route patterns** (a pattern = identical ordered stop sequence). McRAPTOR needs trips of a pattern sorted by departure time at the first stop (and a FIFO-consistent order — if trips overtake each other, split the pattern).
   - `connections`: one row per consecutive stop pair on a trip: `(trip_idx, from_stop, to_stop, dep_time, arr_time, route_idx)` **sorted by dep_time** (useful for tests and A1's time-expanded view).
   - `stop_times` arrays per pattern for fast "earliest trip departing stop *s* at/after time *t*" lookups.
5. Metro feed caveat: if intermediate stop times were *modeled* by the feed author, record `timetable_quality = 'APPROXIMATE'` for metro and state it in the report.
6. Keep each trip's agency/mode so legs can be labeled BUS or METRO.

### 4.4 Choosing between multiple feeds of the same agency
If there are two BMTC feeds: compare calendar coverage, #routes/#trips, stop-coordinate validity, and stop overlap (match stops within 30 m). Prefer the feed whose calendar covers the chosen date and has more valid trips. Keep the other for a one-off stop-overlap cross-check, but build the network from **one** primary feed per agency. Record the choice and why.

### 4.5 Footpaths and access/egress (the multimodal glue)
- **Stop snapping**: snap every stop to the nearest walkable node (KD-tree in the metric CRS); store `snap_dist_m`. Flag stops with `snap_dist_m` above a threshold (default 150 m) as `poorly_snapped`.
- **Footpaths** (stop→stop walking transfers): run a **bounded multi-objective walking search** on the walking graph from each stop's node to nearby stops' nodes within radius `R_transfer` (default 400 m, configurable 300–500 m) and walking time `≤ 7 min`. Walking itself has a time-vs-safety trade-off, so each footpath stores up to **K** Pareto-optimal walking options `(walk_time, walk_dist, walk_risk_exposure)` (default `K = 3`; `K = 1` = fastest only; config key `footpath_options`). Document K in every benchmark result.
- **Access/egress** (origin/destination ↔ stops): computed **on demand per query** with the same bounded walking search (radius `R_access`, default 600 m, configurable), returning up to K options per reachable stop.
- Isolated stops (no footpath at all) are kept but flagged.
- Same-station transfers between bus and metro are just footpaths between nearby stops (no special-casing).

### 4.6 Objectives — exact definitions (all minimized, all additive along a journey)

| Obj | Meaning | How accumulated |
| --- | --- | --- |
| **T** | Travel time in seconds = arrival time at destination − query departure time | Includes waiting, riding, walking. In time-dependent search T is derived from the label's arrival time |
| **C** | Cost in cost-units | Additive per boarding / per leg (Section 4.9). Walking = 0 |
| **X** | Number of transfers | `(number of vehicle boardings) − 1`, floor 0. In McRAPTOR it equals the round index minus 1 |
| **W** | Walking burden | Total walking **distance in metres** (or walking time — pick one, default metres, record it) |
| **E** | Emissions | `Σ distance_km(leg) × emission_factor(mode)`; walking = 0 |
| **S** | Safety-risk exposure | `Σ over walking edges (length_m × safety_risk(edge)) + Σ over transit legs (stop-wait exposure, see below)`. Minimized |

Transit safety exposure (keep it simple and documented): for each boarding stop, wait-time exposure `wait_seconds × stop_risk`, where `stop_risk` = risk of the stop's snapped node/nearby edges; in-vehicle time contributes 0 in v1. State this assumption in the report.

**Objective sets for experiments** (Section 8): `{T,C}` (2), `{T,C,X}` (3), `{T,C,X,W}` (4), `{T,C,X,W,E}` (5), `{T,C,X,W,E,S}` (6). All algorithms receive the objective-set config; unused objectives are simply ignored in dominance.

**Normalization.** Search algorithms A1 and A2 operate on **raw** additive values (dominance needs no normalization). Normalization is needed in three places:
- **B0 and W (during search):** the scalar score must be computable *while searching*, before any reference set exists. So use **fixed global scale constants** `scale_k` per objective, stored in `config/benchmark.yaml`, calibrated **once** from a calibration sample of Core queries (e.g. the 95th percentile of each objective over the fastest-journey results) and then frozen. Score = `Σ w_k · obj_k / scale_k`.
- **Profile re-ranking and preference metrics:** use the same fixed global `scale_k`.
- **Hypervolume and recall tolerances:** per-query min–max over the reference set; fall back to the global `scale_k` if the reference set has a single point.

### 4.7 Safety-risk per edge (builds `S`)
Compute per-edge components from whatever data exists; each normalized to [0,1] (percentile-rank or min–max, record which):

```
lighting_risk(e) = 1 − lighting_coverage(e)             # from streetlight points (buffer ~25 m, tunable)
flood_risk(e)    = normalized flood evidence near e     # KML points: buffer; polygons: intersection
accident_risk(e) = normalized accident evidence near e  # ONLY if the accident file has spatial info
road_risk(e)     = normalized road-design proxy         # signal/crossing density + road class (OSM)
safety_risk(e)   = Σ aₖ · component_k(e) over AVAILABLE components, weights renormalized to sum to 1
```

- Weights live in `config/safety_weights.yaml`; **default = equal weights, labeled `PLACEHOLDER — not calibrated`**.
- Record `component_flags` per edge: which components contributed.
- **Coverage honesty.** If streetlight data covers only some wards/zones, edges outside coverage are *unknown*, not *dark*. For the benchmark the router needs a number everywhere, so: set `lighting_risk` for uncovered edges to the **median of covered edges** and set `s_imputed = True`. Report the share of edges with imputed values.
- **Accident CSV:** if it has lat/lon → snap points to nearest edge (≤ 50 m, tunable) and compute density; else treat as city-level context (save a summary, **do not** assign to edges). Never guess locations.
- **Flood KML:** parse with geopandas (KML driver) or an XML parser if the driver is unavailable; handle multiple layers, folders, and mixed geometry types.
- Night effect: out of scope for the benchmark unless a `time_of_day = night` flag is set in a scenario; if so, `night_multiplier = 1 + β · lighting_risk` with a configurable β (document as MODELED).
- This is a **research composite proxy**, not a measure of real-world safety. Say so in the report.

### 4.8 Static vs dynamic attributes
Keep dynamic effects out of the base graph. Disruptions (Section 7) are an **overlay**: closed edges, trip delays, edge time multipliers, safety penalties. Algorithms accept `Query.disruptions` and apply them without rebuilding the `PreparedNetwork`.

### 4.9 Missing-data fallbacks (all must be labeled in outputs)

| Missing | Fallback |
| --- | --- |
| **Fares** | Use GTFS fare tables if present and consistent. Otherwise use a **PLACEHOLDER cost model in abstract cost-units** (not rupees): bus = `1.0` per boarding, metro = `0.5 + 0.1 × stations travelled`, walking = 0, additive. Write `config/fare_placeholder.csv`, mark `PLACEHOLDER`. Real fares don't change *which algorithm is better* but do change absolute numbers — say so. |
| **Emission factors** | Create `config/emission_factors.csv` with columns `mode, factor, source, status`. Leave `factor`/`source` for the human (`TODO_HUMAN_MUST_FILL`). Until filled, the benchmark uses **relative placeholder factors** (WALK = 0, METRO = 0.5, BUS = 1.0, in relative units) flagged `PLACEHOLDER_RELATIVE_UNITS` — not real-world numbers. |
| **Traffic** | Bus/metro use timetables as given (no traffic multiplier). Road traffic isn't modeled (no road legs in v1). |
| **Streetlights missing** | Use OSM `street_lamp` nodes if available; else drop the lighting component. |
| **Flood missing** | Drop the flood component. |
| **Accident missing / aggregate** | Drop the component. |
| **All safety layers missing** | `S` falls back to the road-design proxy only; if even that fails, run experiments with 5 objectives and say so. |
| **One transit feed missing** | Run on the other plus walking; report the limitation. |
| **Both transit feeds missing** | Blocker for this task — report and stop; don't invent a timetable. |
| **OSM unreadable** | Blocker — report and stop. |

### 4.10 Prepared-network sizes and benchmark regions
The full-city walking graph may have hundreds of thousands of nodes; exact multi-objective search in pure Python can blow up, so A1 is restricted to the Core region. Define three **scenarios** (config in `config/benchmark.yaml`):

| Scenario | Region | Purpose |
| --- | --- | --- |
| **Core** | Disc of ~3 km radius around the area with the highest transit-stop density (auto-computed) | Small enough that A1 finishes → provides an **exact reference**. **A1 runs here only** |
| **Mid** | ~8 km radius | Scaling and moderate-difficulty comparison for A2, W, B0 (reference = best-known; see 8.2) |
| **City** | Whole prepared Bengaluru region | Realism and timeouts for A2, W, B0; reference is "best-known" (see Section 8.2) |

Document the radii actually used and the node/edge/stop counts per scenario.

---

## 5. Algorithm specifications

### 5.1 Common types and interface

```python
@dataclass
class Query:
    origin: tuple[float, float]       # lat, lon (snapped to walking graph)
    destination: tuple[float, float]
    departure_time: int               # seconds after midnight of service date
    objectives: tuple[str, ...]       # subset of ("T","C","X","W","E","S"), e.g. all six
    profile_weights: dict | None      # only used by B0 / for profile re-ranking
    disruptions: "DisruptionSet | None" = None
    start_state: "JourneyState | None" = None   # for rerouting; None = start at origin
    max_transfers: int = 4
    timeout_s: float = 60.0

@dataclass
class Leg:
    mode: str               # WALK | BUS | METRO
    from_ref: str; to_ref: str          # node or stop ids
    dep_time: int; arr_time: int
    segment_ids: list[str]  # walking edge ids, or "trip_id:stop_a>stop_b" per hop (for shared-segment metrics)
    vector: dict            # this leg's contribution to T/C/X/W/E/S

@dataclass
class Journey:
    legs: list[Leg]
    vector: dict            # total J, recomputed from legs (see validation)

@dataclass
class RouteResult:
    journeys: list[Journey]             # the returned (ideally non-dominated) set
    stats: dict                         # runtime_s, peak_mem_mb, labels_generated, labels_expanded,
                                        # rounds, timed_out, label_cap_hit, exact (bool), notes
class Router(ABC):
    name: str
    def prepare(self, net: PreparedNetwork, cfg: dict) -> dict: ...   # one-off preprocessing; returns {time_s, storage_mb}
    def route(self, q: Query) -> RouteResult: ...
```

**Rules for all routers**
- A1, A2 and W return a **valid Pareto set** over `q.objectives` (no returned journey dominates another). B0 returns one journey per profile instead — that's expected and exempt from the non-dominated check.
- Reconstruct full legs with times and recompute `vector` independently from the legs (one shared function in `objectives.py`) — never trust the search's running totals when reporting.
- Respect `timeout_s` and a `label_cap` (config). If either is hit, set `timed_out` / `label_cap_hit` and `exact = False`; return what you have.
- Respect `disruptions` and `start_state`.
- Use the shared dominance code in `pareto.py`; don't re-implement dominance per algorithm.

### 5.2 A1 — Exact multi-objective label-setting search (reference, Core only)
**Idea.** Extend Dijkstra to vector labels. Each node keeps a Pareto set of labels; a label is `(arrival_time, other cost components, parent pointer, last mode, trips taken)`. Process labels in lexicographic order of a priority (e.g. arrival time, then the other objectives); a new label is discarded if an existing label at the same node **dominates** it.

**Graph.** Time-dependent multimodal graph:
- Walking edges: static cost vector.
- Transit: from a stop at time *t*, board the **earliest feasible trip of each pattern** departing ≥ *t* (+ min transfer/boarding buffer, config default 0–60 s) and ride to later stops. Each (stop, trip) boarding is a label with `vehicle = trip_idx` so staying on the vehicle costs no transfer.
- A label's `X` increments when boarding a *different* trip than the previous one.

**Dominance.** Label *a* dominates *b* at a node iff `a.arrival ≤ b.arrival`, and for every selected objective `a.obj ≤ b.obj`, with at least one strict. Be careful that "arrival time" is both the time dimension and objective `T` (don't double count). Two labels at a transit node must also be comparable in *vehicle state* (staying on a vehicle vs having alighted) — keep a state flag or model "on-vehicle" nodes separately.

**Pruning.** (1) **Target pruning** — drop labels dominated by any label already at the destination. (2) Optional **A\*-style lower bounds**: admissible per-objective bounds from the straight-line distance (T: distance / max speed; W, E, S, C, X: 0 unless a safe bound exists). Mark the heuristic as optional and benchmark with/without if time permits.

**Scope.** A1 is run **only on the Core scenario** (Section 4.10). Do not run it on Mid/City in the benchmark (an optional single exploratory run on Mid is allowed, clearly labeled, if the human asks).

**Exactness.** If it finishes with no cap/timeout hit, it is exact → this is the **reference Pareto set** for Core. Keep a `label_cap` (default 2 M labels; tune it so a query stays within a few minutes) and `timeout_s`. When the cap or timeout is hit, set `exact = False`, keep the partial set, and exclude that query from the "exact reference" statistics (it falls back to `BEST_KNOWN`, Section 8.2). Also record labels generated vs. number of objectives — this is where A1's blow-up shows up.

### 5.3 A2 — McRAPTOR-style multi-criteria round-based search
**Idea.** Work on the timetable. Round *k* computes best journeys using exactly *k* vehicle trips; **X is the round index**, so it never needs to be stored in the label.
- Labels per stop per round: a **Pareto bag** of `(arrival_time, C, W, E, S)` (plus parent pointers).
- Round 0: from the origin, apply **access** walking options (up to K per stop) to seed stops.
- Round k: (a) for each route pattern that serves a stop improved in round k−1, scan the pattern in stop order, keeping a bag of boardable trips: at each stop, (i) update stop bags with labels carried on the current trip, (ii) merge in labels from the previous round to board earlier/cheaper trips; (b) relax **footpaths** from improved stops (K options each).
- Stop after `max_transfers + 1` rounds or when no bag improves.
- **Target pruning:** bags at the destination (through **egress** walking from stops) prune labels everywhere.
- Collect destination labels from all rounds and take the global Pareto filter over `(T, C, X, W, E, S)`.
- Walking-only journey (origin→destination) is added as a candidate.
- Note the structural limit: walking on the **street graph** only enters through access, egress, and precomputed footpaths — document it; it's the algorithm's nature.

### 5.4 B0 — Weighted-sum A\* baseline
**Idea.** Collapse the six objectives into one score with a profile's weights and find the single best journey: `score = Σ w_k · obj_k / scale_k` (fixed global `scale_k`, Section 4.6). Returns **one** journey per profile in `config/profiles.csv` (and, when called by W, one journey per supplied weight vector).

**Search.** Time-dependent A\* on the same multimodal graph as A1 (walking edges + timetable boarding), with the same footpath/access/egress options. Two details matter for correctness:
- **State:** `(node_or_stop, current_vehicle_or_none)`, so transfers (`X`) are counted correctly.
- **Time-dependence:** with timetables, the label with the lowest score at a node is not always the best for later boarding (an earlier arrival can catch an earlier vehicle). So keep, per state, the labels that are non-dominated in the pair **(arrival_time, scalar_score)**, not just a single best score. This keeps B0 correct and is still far cheaper than the full 6-objective search.

**Heuristic.** Admissible lower bound on the *remaining scalar score* = `w_T · (straight_line_distance / max_speed) / scale_T`, all other terms 0. `max_speed` = the maximum speed that actually occurs in the prepared network (compute it; cap with a config value). Must never overestimate (test it).

**What it shows.** B0 gives the optimum for the weights it is given, but only **one** route per profile: no alternatives, no view of the trade-off curve. This is the baseline against which W, A2 and A1 are compared; it is not itself a Pareto method.

### 5.5 W — Weight-sweep (approximate Pareto set from many B0 runs)
**Idea.** Run B0 for many different weight vectors, collect the resulting journeys, remove duplicates, and Pareto-filter them. A few lines on top of B0; no new search code.

**Weight vectors** (over the selected objectives only; unselected objectives get weight 0):
1. The unit vectors (one per selected objective) — the extreme single-objective routes.
2. The 5 profiles from `config/profiles.csv` (restricted/renormalized to the selected objectives).
3. `N − (1 + #objectives + 5)` additional vectors drawn from a **seeded Dirichlet(1,…,1)** over the selected objectives. Default `N = 50` (config key `sweep_size`). For 2–3 objectives a regular simplex grid may be used instead.

**Procedure.** For each weight vector call B0 (reusing prepared data and the same heuristic); dedupe journeys by their `segment_ids`; run the shared Pareto filter; return the survivors. `stats` must include `weight_vectors_run`, `distinct_journeys`, `total_runtime_s` and `mean_runtime_per_weight_s`.

**Known limitation (part of the result, not a bug).** A weighted sum can only reach **supported** Pareto points — those on the convex hull of the trade-off. Balanced "compromise" journeys that sit in a concave part of the front are **never** returned, no matter how many weight vectors are tried. The benchmark must therefore (a) report W's recall/hypervolume vs. the reference, and (b) include the non-linear preference test in Section 8.1 (Chebyshev regret), which is where this limitation shows.

**Sensitivity experiment (Core, small query set):** repeat W with `sweep_size ∈ {10, 50, 200}` and report recall and runtime — the cost/quality trade-off of the sweep.

### 5.6 Not implemented: ML-TZSA and NSGA-II
These are **literature-only** in this task. Do **not** build them. In `reports/ALGO_COMPARISON.md` include a short comparison table for them (idea, strengths, what they require — ML-TZSA needs a trained travel-time predictor and zone preprocessing; NSGA-II needs a route encoding with feasibility-preserving mutation — and why they were left out), and list them under future work.

---

## 6. Fairness rules for implementation

### 6.1 Parity checklist (the agent must satisfy and document)
- [ ] All routers consume the same `PreparedNetwork`, objective definitions, K (footpath options), `R_access`, `R_transfer`, min-transfer buffer, `max_transfers`, `label_cap` and `timeout_s`.
- [ ] All routers use the shared `pareto.py`, and B0/W use the same fixed global `scale_k` for scoring.
- [ ] W is reported **both** as total runtime for the whole sweep and per weight vector, so it is not compared unfairly against a single-run algorithm.
- [ ] A1 is only run on Core; every table that includes A1 must say so.
- [ ] All routers are implemented in the same language with a **comparable optimization level** (pure Python + numpy; if numba is used for one router's hot loop, offer it for all or run everything with and without it, and report).
- [ ] Single-threaded, same machine, same Python version; record versions, CPU, RAM in the results.
- [ ] Each query runs in a fresh result object; no cross-query caching beyond what `prepare()` builds (and `prepare` time/storage is reported separately).
- [ ] Warm-up run per router before timing; N repetitions per query (default 3), report median.
- [ ] Seeds fixed for OD sampling, disruption selection, and any stochastic algorithm.

---

## 7. Disruptions and rerouting

### 7.1 Disruption overlay (synthetic, reproducible)

```python
@dataclass
class Disruption:
    event_id: str
    event_type: str      # ROAD_CLOSURE | BUS_DELAY | METRO_DELAY | FLOOD | LIGHTING_OUTAGE | ACCIDENT | TRAFFIC_SPIKE
    start_time: int; end_time: int
    edge_ids: list[int] | None; trip_ids: list[int] | None; stop_ids: list[int] | None
    time_penalty_s: int = 0; closure: bool = False; safety_penalty: float = 0.0
    source_type: str = "SYNTHETIC"; seed: int = 0; generator_version: str = "v1"
```

For this task use: **walking-edge closure/flood** (edge unavailable or penalized), **bus delay** (+5/+10/+20 min on a trip from a given stop onward), **metro delay**, **lighting outage** (raises safety penalty on affected walking edges). Every record is tagged `SYNTHETIC` with a seed. Validate `start < end` and that referenced edges/trips exist; reject otherwise.

### 7.2 Choosing events
Prefer edges/trips that lie **on the baseline best route** of a query (so the event happens mid-journey), but also sample off-route events. Activation point = a point reached ~15–25 min (configurable) after departure, defined by `JourneyState`.

### 7.3 Rerouting protocol
`JourneyState = (current_position (stop or node), current_time, current_mode, completed_legs)`. To reroute:
1. Compute the baseline result for the query.
2. Pick one journey (e.g. the best by a given profile) as the "followed" journey; advance it to the activation time to get the `JourneyState`.
3. Apply the disruption overlay.
4. Call `route(Query(start_state=state, disruptions=...))` for every algorithm and measure **latency** and the quality of the new Pareto set against the exact post-disruption reference where available.
5. Record whether each profile's preferred route changed, and shared-segment ratio between pre/post routes (used by the wider PUMO study).

Limit this experiment to the **Core** and **Mid** scenarios in the time available.

---

## 8. Benchmark design

### 8.1 Metrics (per algorithm, per scenario, per objective-set)
B0 returns one journey per profile, so only the runtime, memory, failure, linear-regret and Chebyshev-regret metrics apply to it; recall, dominated fraction and hypervolume apply to A1, A2 and W.

| Metric | Definition |
| --- | --- |
| Query runtime | median and p95 seconds over queries (after warm-up, over repetitions) |
| Peak memory | peak RSS (psutil) or `tracemalloc` peak during the query |
| Preprocessing | `prepare()` time and storage size |
| Labels generated / expanded | counted inside each algorithm |
| #journeys returned | size of the returned set |
| **Recall** | share of reference-Pareto vectors that are matched by a returned vector within tolerance ε (default 1 % relative per objective; configurable) |
| **Dominated fraction** | share of returned journeys dominated by some reference journey (sanity: should be ~0 for exact methods, may be > 0 for heuristics) |
| **Hypervolume ratio** | HV(returned) / HV(reference) on normalized vectors. Exact HV for ≤ 4 objectives; for 5–6 use a **seeded Monte-Carlo estimate** (default 100 k samples) and say so |
| **Linear profile regret** | For each profile in `config/profiles.csv`: `(best weighted-sum score in returned set − best weighted-sum score in reference set) / reference best`, averaged over queries. Includes B0. **Expect ≈ 0 for every method that finds the weighted optimum** (a weighted-sum optimum always lies on the Pareto front), so this is a sanity check, not a discriminator |
| **Chebyshev regret** (non-linear preference) | For each profile: score each journey by `max_k (w_k · obj_k / scale_k)` (the "worst weighted objective", which favors balanced compromises); regret = `(best in returned set − best in reference set) / reference best`, averaged over queries. This is where **W and B0 are expected to lose**, because Chebyshev optima can sit on unsupported Pareto points |
| **Failure rate** | share of queries with timeout, label-cap hit, or no journey found |
| **Scaling** | median runtime and #labels vs. number of objectives (2 → 6) for A1 (Core), A2, W; and vs. scenario size (Core → Mid → City) for A2, W, B0 |
| **Disruption** | rerouting latency, post-disruption recall and regret |

### 8.2 Reference set ("ground truth")
- **Core:** exact A1 Pareto set, valid only when `exact=True` (no cap/timeout). Queries where A1 hit its cap are marked `reference_type = BEST_KNOWN`.
- **Mid and City (A1 is not run):** reference = **non-dominated union of the outputs of W, A2 and the B0 journeys** for that query, flagged `reference_type = BEST_KNOWN`. Because the reference is built from the compared methods themselves, recall against it is **optimistic** for those methods — state this clearly and **never** call it the true Pareto set.
- Always store `reference_type` per query row. Report Core (exact) results and Mid/City (best-known) results in separate tables.

### 8.3 Query set
- Seeded generation, **stratified** by: straight-line distance band (short ≈ 1–3 km, medium 3–8 km, long > 8 km), **transit richness** (stops within `R_access` of origin and destination: low / high), and **departure band** (morning peak, midday, evening peak, late evening — the exact hours come from what the timetable actually has).
- Defaults: 10 queries per cell for Core, 6 for Mid, 4 for City — reduce if the runtime budget requires, and report the final counts.
- Write queries to `data/benchmark/queries_<scenario>.parquet` so reruns are identical.
- Include a few hand-picked queries along a metro line to sanity-check against the published timetable.

### 8.4 Pre-registered decision rule (write to `docs/BENCHMARK_PROTOCOL.md` and commit before any run)
Compared **Pareto-set methods**: **W** and **A2**. **A1** is the exact reference (judged on Core only). **B0** is the single-route baseline. Defaults — the human may edit them *before* the run:
1. **Quality gate (Core, exact reference, 6 objectives):** a method is **eligible** if median hypervolume ratio ≥ **0.95** *and* mean Chebyshev regret ≤ **5 %**. Linear regret is reported as a sanity check but is **not** used in the gate (see 8.1).
2. Repeat the same gate on **Mid** against the best-known reference and report it separately, flagged as optimistic.
3. Among eligible methods, the winner has the **lowest p95 runtime** on Mid + City (for W use total sweep runtime).
4. Tie-breaks (runtime within 10 %): lower peak memory, then lower preprocessing cost.
5. If **no** method is eligible, report the best on quality and say explicitly that none met the gate.
6. **A1** passes the gate by definition on Core; report its Core runtime, label counts and objective-scaling curve, and say plainly whether it is practical only as an offline reference tool. **B0** is reported as the baseline and is not eligible to "win".
7. A separate **qualitative note** (not part of the rule): customizability — how easily each method accepts new objectives, disruptions, and personalization.

### 8.5 Outputs
- `results/raw_runs.csv` — one row per (algorithm, scenario, query, objective-set, repetition) with all stats.
- `results/summary_<scenario>.csv` — aggregated metrics.
- Figures (matplotlib, saved to `reports/figures/`): runtime vs objectives; recall vs objectives; hypervolume ratio; label counts; runtime distribution by distance band; Pareto-set size; disruption latency.
- `reports/ALGO_COMPARISON.md`: dataset description (counts, coverage, imputation shares, placeholders used), implementation notes and deviations from the papers, tables, figures, conclusion per the protocol, limitations.

---

## 9. Testing and validation

### 9.1 Unit tests (`pytest`)
- **Dominance / Pareto filter:** hand-made vectors with ties, duplicates, strict/non-strict dominance; ε-dominance matching.
- **GTFS time parsing:** `24:30:00`, `25:00:00` handled; day rollover never wraps.
- **Objective calculators:** a fixed toy journey with known T, C, X, W, E, S.
- **Hypervolume:** known 2-D and 3-D cases against hand calculations; Monte-Carlo estimate within tolerance of exact on small cases.
- **Footpath/option generation:** on a tiny graph, K options are mutually non-dominated.
- **Config/profiles:** each profile's weights sum to 1.0.
- **CRS:** metric operations only in EPSG:32643 (assert on distance helpers).

### 9.2 Algorithm-correctness tests (the important ones)
1. **Toy network with a hand-computed Pareto set** (≈ 8–10 stops, 2 bus routes, 1 metro line, a small walking graph with a short-risky vs long-safe alternative). A1 must reproduce the hand-computed set exactly; A2 must too; W must return a non-dominated subset of it that contains the extreme (single-objective-optimal) journeys; B0 must return the correct weighted-sum optimum for each profile. Design the toy network so that at least one balanced journey is a Pareto point that **no** weight vector can select (an unsupported point) — W must miss it while A1 and A2 find it.
2. **Single-objective agreement:** with `objectives=("T",)` all of A1, A2, B0 (weights = T only) and W must return the **same earliest-arrival time** on many random queries.
3. **Brute-force oracle on tiny random networks** (use `hypothesis` or seeded generators): enumerate all feasible journeys (≤ 3 trips) by exhaustive search, take the Pareto set, compare with A1 and A2 (must match exactly). Also check that B0's result for random weights equals the exhaustive weighted-sum optimum, and that every journey W returns is a member of the exhaustive Pareto set.
4. **No dominated outputs:** for every router and query, assert the returned set contains no dominated journey (B0 exempt; W's output is Pareto-filtered by construction, so it must pass).
5. **Feasibility validator** (shared function): legs connect end-to-start, times are monotone, boarding time ≥ arrival at stop + min transfer buffer, trip actually serves the stops used, no closed edge used, vectors recomputed from legs equal the reported vector (tolerance 1e-6).
6. **Determinism:** same seed → byte-identical results.
7. **Disruption tests:** a closed edge is never used; a delayed trip's arrival times shift correctly; rerouting from a mid-journey state never starts before `current_time`.
8. **Heuristic admissibility (B0/W):** on random queries the A\* heuristic never exceeds the true remaining scalar cost found by an exhaustive/Dijkstra run.
9. **Monotonicity:** dropping an objective can never *increase* the size of the optimal Pareto set's T-minimum (the fastest time is unchanged).

### 9.3 Data validation (real data)
- Coverage report: #nodes/edges, largest-component share, #stops per mode, % stops poorly snapped, % edges with each safety component, % edges imputed, number of footpaths, K used.
- Spot-check **10 queries** on real data: print the returned journeys in readable form (legs, times, stops) so the human can eyeball them. Include metro-only queries and compare travel times to the published metro timetable (expect agreement only within the feed's approximation).
- Cross-feed check: if two BMTC feeds exist, report stop-overlap and route-overlap.
- Sensitivity run on a small query set: K = 1 vs 3, `R_access` 400 vs 600 m — show how results shift, so the human understands the parameters.

### 9.4 Reproducibility
- A single command per phase (Makefile or CLI), e.g. `make data`, `make test`, `make bench-core`, `make bench-all`, `make report`.
- Seeds, config snapshot, package versions, and git commit hash are saved with every results file.

---

## 10. Implementation guidance

- **Python ≥ 3.10.** Suggested libraries: `numpy`, `pandas`, `pyarrow`, `geopandas`, `shapely`, `pyproj`, `scipy` (cKDTree), `pyyaml`, `psutil`, `matplotlib`, `pytest`, `hypothesis`; PBF reading via `pyrosm` or `osmium` (pick one, note it in `DECISIONS.md`); KML via geopandas/fiona (enable the KML driver) or manual XML parsing. `networkx` is allowed for tests/offline utilities only. `numba` is optional and subject to the parity rule in Section 6.1.
- **Data structures.** CSR arrays for the walking graph; per-pattern arrays of stop sequences and trip departure/arrival matrices for transit; dictionary/hash maps only outside hot loops.
- **Dominance performance.** Keep Pareto bags as small numpy arrays or sorted lists; dominance check is the hot path in every algorithm — implement once, test heavily, then optimize.
- **Memory.** Use compact dtypes (int32, float32 where safe); store parent pointers as arrays of ints.
- **Label limits.** Always expose `label_cap` and `timeout_s`; never allow an infinite run.
- **Logging.** Use `logging`, not `print`, inside the library; the CLI prints summaries.
- **Don't over-engineer.** This is a research prototype; prefer clear code, docstrings that reference the paper's idea, and tests.

---

## 11. Phases and acceptance criteria

| Phase | Work | Done when |
| --- | --- | --- |
| **P0** | Inventory `Data/`; write `INVENTORY.md`; confirm what feeds/layers exist | Inventory printed; classification table shown |
| **P1** | Preprocessing → `PreparedNetwork` (walking graph, transit timetable, footpaths, safety components, placeholders) | Network loads from cache; coverage report generated; data tests pass |
| **P2** | `pareto.py`, `objectives.py`, validators, toy network + oracle tests | Unit tests pass |
| **P3** | A1 label-setting (Core only, with label cap and timeout) | Toy/oracle tests pass; Core queries run; exact references produced |
| **P4** | B0 weighted A\* and W weight-sweep | B0 matches the exhaustive weighted optimum on toy/oracle tests; W returns a non-dominated subset of A1's set on Core; sweep-size experiment runs |
| **P5** | A2 McRAPTOR-style | Matches A1 on toy/oracle tests; Core runs; Mid and City queries run within the timeout |
| **P6** | Benchmark runner, query generation, metrics, disruptions | `make bench-core` produces `raw_runs.csv` and summaries |
| **P7** | Write `docs/BENCHMARK_PROTOCOL.md` **(before the real runs)**, run Core → Mid → City, scaling and disruption experiments | Results CSVs and figures exist |
| **P8** | Reports | `ALGO_COMPARISON.md` and `SLIDE_SUMMARY.md` written, following the pre-set decision rule, with limitations |

At the end of each phase: run `pytest`, commit with a clear message, print status.

---

## 12. Things to state honestly in every report

- OSM-derived walking times are modeled; metro intermediate stop times may be approximate; bus feeds may be unofficial or outdated.
- Fares and emission factors are **placeholders** unless the human supplied sourced values; absolute numbers for C and E are not real-world values.
- The safety score is a composite proxy built from the layers that were actually available; imputed share must be reported; accident data is used only if spatial.
- Weights (profiles, safety components) and disruptions are project parameters, not observed facts.
- W can only find supported Pareto points; its gaps (shown by recall, hypervolume and Chebyshev regret) are an inherent property of weighted sums, not an implementation fault.
- A1 was run only on the Core region with a label cap; Mid/City references are best-known sets built from the compared methods and are optimistic for them.
- ML-TZSA and NSGA-II were reviewed in the literature but not implemented.
- Linear profile regret is ≈ 0 for any method that finds the weighted optimum, so it does not discriminate between methods; the Chebyshev regret, recall and hypervolume do.
- Where the exact reference was unavailable, recall is against a best-known set.
- Results hold for this dataset, these queries, and this implementation, not for algorithms in general.

---

## 13. First prompts the human may give you

- "Read `CONTEXT.md` and `ALGO_TASK.md`, then run Phase P0 (inventory of `Data/`) and show me what you found."
- "Run P1: build the PreparedNetwork from whatever is available and show the coverage report."
- "Implement P2–P3 (Pareto utilities, tests, A1 on Core) and show me the toy-network test results."
- "Implement B0 and W (P4), then A2 (P5), and run the Core benchmark."
- "Write the benchmark protocol, then run all scenarios and generate the report."

Start with P0. Be concise in chat; put detail into the generated docs.
