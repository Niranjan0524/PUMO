#!/usr/bin/env python3
"""Inventory raw task inputs without modifying them.

Run from any directory: python scripts/00_inventory.py
The script writes data/INVENTORY.md and data/inventory.json next to this file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
# On Windows, ``data`` aliases the user's existing raw ``Data`` folder.
# Keep all generated outputs separate so raw inputs remain untouched.
OUT = ROOT / "derived_data"
GTFS_REQUIRED = {"stops.txt", "routes.txt", "trips.txt", "stop_times.txt"}
LAT_NAMES = {"lat", "latitude", "stop_lat", "y"}
LON_NAMES = {"lon", "lng", "longitude", "stop_lon", "x"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        yield from csv.DictReader(f)


def csv_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        counts = Counter()
        samples = []
        inferred = {c: "unknown" for c in columns}
        for row in reader:
            counts.update({c: 1 for c in columns if row.get(c, "").strip() == ""})
            if len(samples) < 5:
                samples.append(row)
            for c in columns:
                value = (row.get(c) or "").strip()
                if not value or inferred[c] == "string":
                    continue
                if re.fullmatch(r"[-+]?\d+", value):
                    if inferred[c] == "unknown": inferred[c] = "integer"
                elif re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", value):
                    inferred[c] = "number"
                else:
                    inferred[c] = "string"
        total = reader.line_num - 1
    normalized = {c.lower().strip() for c in columns}
    has_lat = bool(normalized & LAT_NAMES)
    has_lon = bool(normalized & LON_NAMES)
    location = any("location" in c.lower() for c in columns)
    return {"row_count": total, "columns": columns, "dtypes_inferred": inferred,
            "null_percent": {c: round(100 * counts[c] / total, 2) if total else None for c in columns},
            "sample_rows": samples, "latitude_like": has_lat, "longitude_like": has_lon,
            "combined_location_like": location}


def gtfs_summary(folder: Path) -> dict[str, Any]:
    names = {p.name.lower() for p in folder.iterdir() if p.is_file()} if folder.is_dir() else set()
    if folder.is_file() and zipfile.is_zipfile(folder):
        with zipfile.ZipFile(folder) as zf:
            names = {Path(n).name.lower() for n in zf.namelist()}
    if not GTFS_REQUIRED.issubset(names):
        return {}
    def table(name: str) -> list[dict[str, str]]:
        p = folder / name
        if p.is_file():
            return list(rows(p))
        with zipfile.ZipFile(folder) as zf:
            matches = [n for n in zf.namelist() if Path(n).name.lower() == name]
            if not matches: return []
            import io
            return list(csv.DictReader(io.TextIOWrapper(zf.open(matches[0]), encoding="utf-8-sig", errors="replace")))
    agencies = table("agency.txt")
    routes = table("routes.txt")
    calendar = table("calendar.txt")
    agency_text = " ".join((r.get("agency_name", "") for r in agencies)).lower()
    route_text = " ".join((r.get("route_long_name", "") + " " + r.get("route_short_name", "") for r in routes)).lower()
    is_metro = any(token in agency_text + " " + route_text for token in ("bmrcl", "namma metro", "purple line", "green line"))
    mode = "METRO" if is_metro else "BUS"
    dates = [r.get("start_date", "") for r in calendar] + [r.get("end_date", "") for r in calendar]
    dates = sorted(x for x in dates if x)
    return {"classification": mode, "files_present": sorted(names),
            "required_files_missing": sorted(GTFS_REQUIRED - names),
            "agency_names": [r.get("agency_name", "") for r in agencies],
            "route_name_samples": [r.get("route_long_name") or r.get("route_short_name") for r in routes[:10]],
            "counts": {"stops": len(table("stops.txt")), "routes": len(routes),
                       "trips": len(table("trips.txt")), "stop_times": len(table("stop_times.txt"))},
            "calendar_date_range": [dates[0], dates[-1]] if dates else None,
            "fare_files_present": sorted(n for n in names if n.startswith("fare_"))}


def kml_summary(path: Path) -> dict[str, Any]:
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    root = ET.parse(path).getroot()
    layers: dict[str, Counter] = {}
    boxes = []
    for folder in root.findall(".//k:Folder", ns):
        name = folder.findtext("k:name", default="(unnamed)", namespaces=ns)
        layers.setdefault(name, Counter())
    for placemark in root.findall(".//k:Placemark", ns):
        parents = [e.findtext("k:name", default="(unnamed)", namespaces=ns) for e in root.findall(".//k:Folder", ns)]
        layer = parents[-1] if parents else "(root)"
        layer = layer if layer in layers else "(root)"
        types = [tag.rsplit("}", 1)[-1] for tag in (c.tag for c in placemark) if tag.endswith(("Point", "LineString", "Polygon", "MultiGeometry"))]
        layers.setdefault(layer, Counter()).update(types or ["unknown"])
        for coord in placemark.findall(".//k:coordinates", ns):
            for item in (coord.text or "").split():
                parts = item.split(",")
                if len(parts) >= 2:
                    try: boxes.append((float(parts[0]), float(parts[1])))
                    except ValueError: pass
    return {"layers": {name: dict(counts) for name, counts in layers.items()},
            "feature_count": sum(sum(c.values()) for c in layers.values()),
            "geometry_types": sorted({g for c in layers.values() for g in c}),
            "bbox_lonlat": ([min(x for x, _ in boxes), min(y for _, y in boxes), max(x for x, _ in boxes), max(y for _, y in boxes)] if boxes else None),
            "attribute_fields": sorted({d.get("name", "") for d in root.findall(".//k:Data", ns)})}


def pbf_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"size_bytes": path.stat().st_size, "bbox": None}
    osmium = shutil.which("osmium")
    if osmium:
        try:
            proc = subprocess.run([osmium, "fileinfo", "-e", str(path)], text=True, capture_output=True, timeout=180)
            result["fileinfo"] = (proc.stdout + proc.stderr).strip()[-4000:]
            result["readable_by_osmium"] = proc.returncode == 0
            match = re.search(r"([-d.]+),([-d.]+),([-d.]+),([-d.]+)", result["fileinfo"])
            if match: result["bbox"] = [float(v) for v in match.groups()]
        except Exception as exc: result["readability_error"] = str(exc)
    else:
        result["readable_by_osmium"] = None
        result["note"] = "osmium CLI not installed; PBF header not parsed"
    return result


def inspect(path: Path) -> dict[str, Any]:
    rel = path.relative_to(ROOT).as_posix()
    entry: dict[str, Any] = {"path": rel, "name": path.name, "size_bytes": path.stat().st_size,
                             "extension": "".join(path.suffixes).lower(), "sha256": sha256(path), "status": "found"}
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv" or suffix == ".txt":
            entry["type"] = "GTFS_TABLE" if path.parent.name.lower() in {"bmtc", "bmrcl"} else "TABULAR"
            entry["summary"] = csv_summary(path)
        elif suffix in {".kml"}:
            entry["type"] = "KML"; entry["summary"] = kml_summary(path)
        elif path.name.lower().endswith(".osm.pbf"):
            entry["type"] = "OSM_PBF"; entry["summary"] = pbf_summary(path)
        elif suffix == ".zip" and zipfile.is_zipfile(path):
            entry["summary"] = gtfs_summary(path)
            entry["type"] = "GTFS" if entry["summary"] else "ZIP"
        else:
            entry["type"] = "UNCLASSIFIED"
    except Exception as exc:
        entry["status"] = "unreadable"; entry["error"] = f"{type(exc).__name__}: {exc}"
        entry.setdefault("type", "UNKNOWN")
    return entry


def main() -> int:
    if not DATA.is_dir():
        print(f"Data directory not found: {DATA}", file=sys.stderr); return 2
    files = sorted(p for p in DATA.rglob("*") if p.is_file())
    entries = [inspect(path) for path in files]
    # Feed summaries are attached to each detected GTFS folder to distinguish them by contents.
    for feed in (DATA / "bmtc", DATA / "bmrcl"):
        if feed.is_dir():
            try:
                summary = gtfs_summary(feed)
                target = next((e for e in entries if e["path"] == feed.relative_to(ROOT).as_posix() + "/agency.txt"), None)
                if target is not None: target["feed_summary"] = summary
            except Exception as exc:
                entries.append({"path": feed.relative_to(ROOT).as_posix(), "type": "GTFS", "status": "unreadable", "error": str(exc)})
    OUT.mkdir(exist_ok=True)
    doc = {"generated_utc": datetime.now(timezone.utc).isoformat(), "data_root": "Data/", "files": entries}
    (OUT / "inventory.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = Counter(e["status"] for e in entries)
    unclassified = [e for e in entries if e.get("type") in {"UNCLASSIFIED", "UNKNOWN"}]
    lines = ["# Data inventory", "", f"Generated: {doc['generated_utc']}", "", "Raw inputs are read-only. Checksums are SHA-256.", "",
             "## Summary", "", "| Status | Files |", "|---|---:|"]
    for status in ("found", "unreadable"):
        lines.append(f"| {status} | {counts[status]} |")
    lines.append(f"| unclassified | {len(unclassified)} |")
    lines += ["", "## Files", "", "| Path | Type | Size (bytes) | Status | SHA-256 |", "|---|---|---:|---|---|"]
    for e in entries:
        lines.append(f"| `{e['path']}` | {e.get('type','UNKNOWN')} | {e.get('size_bytes','')} | {e['status']} | `{e.get('sha256','')}` |")
        if e.get("error"): lines.append(f"\n  - Read error: {e['error']}")
        summary = e.get("feed_summary") or (e.get("summary") if e.get("type") in {"KML", "OSM_PBF"} else None)
        if summary: lines.append(f"\n  - Summary: `{json.dumps(summary, ensure_ascii=False)[:1200]}`")
    (OUT / "INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("| Status | Files |\n|---|---:|")
    print(f"| found | {counts['found']} |\n| unreadable | {counts['unreadable']} |\n| unclassified | {len(unclassified)} |")
    print("\nGTFS feed classifications:")
    for feed in (DATA / "bmtc", DATA / "bmrcl"):
        if feed.is_dir():
            summary = gtfs_summary(feed)
            print(f"- {feed.name}: {summary.get('classification','not a complete GTFS feed')} — {summary.get('counts',{})}")
    print(f"\nWrote {OUT / 'INVENTORY.md'} and {OUT / 'inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
