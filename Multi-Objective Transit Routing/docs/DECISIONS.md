# Implementation decisions

| Date | Decision | Reason / impact |
|---|---|---|
| 2026-10-16 | Store generated files below `derived_data/`, not `data/`. | Windows resolves names case-insensitively and `Data/` is the raw, read-only input directory. Generated files must not alias it. |
| 2026-10-16 | Use `pyrosm` for PBF extraction with the configured Bengaluru bounding box plus buffer. | Keeps the multi-gigabyte regional extract out of the routing graph and avoids custom PBF parsing. |
| 2026-10-16 | Use one common weekday service date inside both feeds' overlap. | Both feed calendars overlap; selected 2026-09-07, the first Monday in their shared coverage. |
| 2026-10-16 | Keep the provided accident table as context only. | It reports state/city junction aggregates and has no coordinates; assigning it to walking edges would fabricate locations. |
| 2026-10-16 | Treat streetlight CSV as ward-level context. | The file has ward identifiers and counts but no geometry; no ward boundary layer is supplied. |
| 2026-10-16 | Use road class and nearby flood points as the available edge safety components. | No geolocated accident or streetlight records were found in the available files. The score is a proxy, not a safety measurement. |
| 2026-10-16 | Calculate walking edge lengths after projecting to EPSG:32643. | Distances in degrees are invalid for metric routing costs. |
| 2026-10-16 | Use the task-specified relative emission factors and placeholder fare model until GTFS fares are verified as consistent. | Absolute costs/emissions must not be presented as observed values. |
| 2026-10-16 | B0 retains the non-dominated (arrival, scalar score) labels per vehicle state. | Earlier arrival can catch a better timetable connection, so a single lowest-score label can discard the weighted optimum. |
| 2026-10-16 | Disable B0's time heuristic when observed edge speed exceeds the configured 40 m/s bound; adjust the metric chord by the minimum edge-length/chord ratio. | A lower speed or an unadjusted chord can overestimate remaining travel time when feed distance and snapped graph endpoints differ. A zero heuristic preserves correctness. |

