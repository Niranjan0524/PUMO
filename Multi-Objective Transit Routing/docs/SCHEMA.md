# Prepared network schema

All generated products live under `derived_data/` to keep the case-insensitive Windows path `Data/` raw and read-only.

- `processed/walking_nodes.parquet`: compact graph node index, original OSM node ID, WGS84 point geometry and city flag.
- `processed/walking_edges.parquet`: directed walk edge IDs/endpoints, OSM way ID, WGS84 geometry, EPSG:32643 length, modeled walk time, road class and safety components.
- `processed/stops.parquet`: feed stop IDs, globally unique integer stop index, agency/mode, coordinates, snapped node, snap distance and poor-snap flag.
- `processed/routes.parquet`, `trips.parquet`, `stop_times.parquet`, `connections.parquet`: filtered common-date timetable with feed IDs retained and integer indices for routing.
- `processed/coverage.json`: input counts, graph coverage, stop snapping, evidence coverage and limitations.
- `quarantine/`: input records rejected with an explicit reason.

Times are integer seconds after midnight and retain values above 86400. Spatial interchange is EPSG:4326; lengths and nearest-neighbor calculations use EPSG:32643. Derived and modeled fields are identified in provenance fields or the coverage and decision documents.

