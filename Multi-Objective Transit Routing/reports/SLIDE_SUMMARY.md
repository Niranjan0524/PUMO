# PUMO routing algorithms — slide summary

- **Question:** which Pareto-set method best fits this Bengaluru multimodal dataset under the registered quality and runtime rule?
- **Compared:** B0 weighted-sum A*, W weight-sweep, A2 McRAPTOR-style rounds, and Core-only exact A1.
- **Measured:** runtime and memory, labels, Pareto recall, hypervolume, linear and Chebyshev regret, failures, and objective scaling.
- **Decision:** Core gate unavailable because A1 produced no exact six-objective references; no winner declared.
- **Interpretation:** All four methods reached timeout or label caps frequently in the pilot; report quality and runtime together. W samples supported fronts only; A1 supplies exact Core references only on uncapped queries.
- **Caveats:** metro times can be approximate; fares/emissions are placeholders; safety is a proxy based on available layers; Mid/City reference sets are optimistic.
