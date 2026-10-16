#!/usr/bin/env python3
"""Render a human-readable data coverage report from generated manifests."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "derived_data"
coverage = json.loads((BASE / "processed" / "coverage.json").read_text(encoding="utf-8"))
inventory = json.loads((BASE / "inventory.json").read_text(encoding="utf-8"))
files = {entry["path"]: entry for entry in inventory["files"]}
def file_summary(path):
    item = files.get(path, {})
    summary = item.get("summary", {})
    return summary

acc = file_summary("Data/Accident data.csv")
street = file_summary("Data/streetlight data.csv")
lines = ["# Data preparation report", "", f"Prepared service date: **{coverage['service_date']}**.",
    "", "Generated routing artifacts are in `derived_data/processed/`; raw `Data/` inputs are unchanged.",
    "", "## Inventory", "", f"- {len(files)} raw files were inventoried; {sum(v.get('status')=='unreadable' for v in files.values())} were unreadable.",
    f"- BMTC: {coverage['transit']['bus_stops']:,} in-region stops; BMRCL: {coverage['transit']['metro_stops']:,} in-region stops.",
    "- Agency classification was based on agency/route contents.", "", "## Prepared layers", "",
    f"- Walking graph: {coverage['walking_graph']['nodes']:,} nodes and {coverage['walking_graph']['directed_edges']:,} directed edges.",
    f"- Transit: {coverage['transit']['trips']:,} active trips, {coverage['transit']['connections']:,} connections, {coverage['transit']['routes']:,} routes.",
    f"- Stop snapping: {coverage['transit']['poorly_snapped_stops']} stops ({coverage['transit']['poorly_snapped_share']:.2%}) exceed 150 m.",
    f"- Footpaths: {coverage.get('footpaths', {}).get('options', 0):,} Pareto options across {coverage.get('footpaths', {}).get('stop_pairs', 0):,} stop pairs (up to K={coverage.get('footpaths', {}).get('K', 3)}).",
    "", "## Evidence and modeled values", "",
    f"- Flood: {coverage['safety']['flood_risk'].get('features', 0)} KML point features; {coverage['safety']['flood_risk'].get('available_edges', 0):,} edges intersect the 25 m proximity threshold.",
    f"- Streetlights: {street.get('row_count', 0)} ward-level count rows, with no coordinates; not assigned to edges.",
    f"- Accidents: {acc.get('row_count', 0)} aggregate rows without coordinates; retained as context only.",
    "- Road-class risk and walking times are MODELED. Flood proximity is DERIVED from the supplied KML.",
    "- No point-level streetlight layer was found in the supplied data; the streetlight component is omitted.",
    "- Fare and emission values use task-specified placeholders; they are not real rupee or emissions estimates.",
    "- Metro intermediate stop times may be approximate. Synthetic disruptions have not yet been generated.",
    "- Safety is a composite proxy from available components and is not a real-world safety measurement.",
    "", "## Reproducibility and caveats", "",
    "- CRS: EPSG:4326 storage, EPSG:32643 metric operations.",
    "- Service date: first weekday in the common BMTC/BMRCL calendar range (2026-09-07).",
    "- Footpaths use bounded multi-objective search; per-node labels are capped at 12 for runtime, so this transfer cache is approximate.",
    "- Coverage details and exact component counts are in `processed/coverage.json`; raw checksums are in `INVENTORY.md`.", ""]
(BASE / "DATA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(BASE / "DATA_REPORT.md")
