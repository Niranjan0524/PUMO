# PUMO multi-objective transit routing

This folder contains the Bengaluru data preparation and routing-algorithm comparison described in `ALGO_TASK.md`. Raw inputs are in `Data/` and are read-only. On Windows, generated files use `derived_data/` because `data/` would alias the raw `Data/` directory.

## Commands

```powershell
python scripts/00_inventory.py
python scripts/01_prepare_network.py
python scripts/02_build_footpaths.py
python -m pytest -q
```

PBF extraction uses `pyrosm`; the generated network uses EPSG:4326 geometries and EPSG:32643 metric distances. See `docs/DECISIONS.md` and `derived_data/DATA_REPORT.md` for data provenance, modeled values, and limitations.
