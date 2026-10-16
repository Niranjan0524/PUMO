# Benchmark protocol

**Status:** pre-registered pilot protocol v1.5. Earlier pilots exposed full-graph speed-bound rescans and Python allocation-tracing overhead; those runs were interrupted and are excluded. A v1.4 harness smoke run confirmed output generation but the five objective tiers made the full stratified pilot impractically long. v1.5 compares time+cost (`TC`) with the full six-objective set (`TCXWES`), preserving the same seeded stratified sample, decision gates, one repetition, shared 5-second per-algorithm query budget, and 10,000-label cap. It samples process resident memory every 10 ms and reports peak RSS increase during each route; runtime excludes sampler shutdown. Results support exploratory comparison only; earlier pilot outputs are not combined with v1.5.

## Comparison scope

- Implemented methods: A1 exact label-setting (Core only), A2 McRAPTOR-style, B0 weighted A*, and W weight-sweep.
- W and A2 are the competing Pareto-set methods. B0 is a single-route baseline. A1 supplies the exact Core reference when it finishes uncapped and within timeout.
- ML-TZSA and NSGA-II are literature-only; they are not benchmarked.
- All methods use the same frozen `PreparedNetwork`, query set, objective definitions, footpath/access settings, and global scales. Single-threaded execution and identical timeout/cap settings are used. Memory is the peak sampled process RSS increase above the pre-route baseline, not Python allocator-only memory.

## Query and repetition policy

- Query generation is seeded (`20261016`) and stratified by distance band, nearby-stop richness, and a departure period supported by timetable departures. Hand-picked metro segment checks are included when available.
- Source query sets retain the original seeded counts (10 per cell for Core, 6 for Mid, and 4 for City). Pilot runs select one query per populated distance/richness/departure cell and retain the hand-picked metro sanity queries; every empty source stratum is reported.
- Each router receives one warm-up query. Pilot timed queries use one repetition; median and p95 are across the seeded query sample and are explicitly exploratory. Profile searches for B0 and all vectors in W share the same 5-second per-algorithm query budget. W reports how many of its configured vectors ran before that budget expired. Every method uses the same 10,000-label cap.
- Objective scaling is sampled at two informative tiers: time+cost (`TC`) and the complete six-objective set (`TCXWES`). The router interfaces accept other user-selected subsets; those can be benchmarked separately with the script's `--objective-sets` option.
- W sweep-size sensitivity is evaluated on one seeded City query from each short/medium/long distance band, using the six-objective set, sweep sizes 10/25/50, and the same one-repetition, 5-second budget and label cap. This targeted sensitivity run is not pooled into the main algorithm comparison.
- The benchmark manifest records source checksums, configuration, Python/package versions, CPU/RAM, seed, and repository revision where available.

## Metrics and references

- Tolerance is 1% relative per objective. Hypervolume uses per-query min/max normalization against the reference set (global scale fallback when the reference has one point); the reference point is 1.1 in every normalized dimension. Exact hypervolume is used through four objectives; 5–6 objectives use 100,000 seeded Monte Carlo samples.
- Linear and Chebyshev profile regret use the same fixed global objective scales used by B0/W. Failure includes timeout, label-cap hit, or no route.
- Core reference: the A1 Pareto set only when A1 finishes with `exact=True`. Otherwise use a non-dominated best-known union and label the query `BEST_KNOWN`.
- Mid/City reference: non-dominated union of W, A2, and all B0 profile journeys. This is optimistic for those methods and is never called ground truth.
- Core exact and Mid/City best-known results are reported separately.

## Winner rule (locked)

1. A Pareto-set method is Core-eligible if its median 6-objective hypervolume ratio is at least 0.95 and mean Chebyshev regret is at most 5% against exact Core references. If no sampled Core query has an exact A1 reference, this gate is unavailable and no overall winner may be declared.
2. Apply the same gate on Mid against its best-known reference, labeled optimistic.
3. Among eligible methods, the winner has the lowest p95 runtime on Mid + City; W uses total sweep runtime.
4. Runtime within 10% is a tie; compare sampled peak RSS increase, then preprocessing cost.
5. If no method qualifies, report the highest-quality method and state that none met the gate.
6. A1 passes the Core gate by definition only on its exact queries. Report its runtime, label growth, objective scaling, and offline practicality separately. B0 cannot win.
7. Customizability for selected objective subsets and preference profiles is a qualitative note, not part of the decision rule. Disruption handling and rerouting are outside this benchmark and reserved for future work.

## Limitations to report

The result applies only to this Bengaluru snapshot, these scenarios and queries, and this implementation. W cannot discover unsupported Pareto points. Mid/City references are optimistic. Modeled walking times, approximate metro times, placeholder fare/emission units, and proxy safety values are not observed real-world measurements.
