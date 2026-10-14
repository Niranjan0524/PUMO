import os
import geopandas as gpd
import pandas as pd

ROADS_FILE = "processed_data/road_edges.gpkg"
FLOOD_FILE = (
    "raw_data/Flood/"
    "6b3c63b0-f461-4e9c-a2c2-006f734c5b41.kml"
)
OUTPUT_FILE = "processed_data/flood_edge_safety.csv"

MATCH_DISTANCE = 30
PROJECTED_CRS = "EPSG:32643"

os.makedirs("processed_data", exist_ok=True)

print("Loading road edges...")
roads = gpd.read_file(ROADS_FILE).reset_index(drop=True)

required = ["u", "v", "key"]
missing = [c for c in required if c not in roads.columns]

if missing:
    raise ValueError(f"Road file is missing columns: {missing}")

roads["edge_id"] = range(len(roads))
roads["u"] = roads["u"].astype("int64")
roads["v"] = roads["v"].astype("int64")
roads["key"] = roads["key"].astype("int64")

print("Loading flood locations...")
flood = gpd.read_file(FLOOD_FILE, driver="KML")

if roads.crs is None or flood.crs is None:
    raise ValueError("Both layers must have a defined CRS.")

roads_m = roads.to_crs(PROJECTED_CRS)
flood_m = flood.to_crs(PROJECTED_CRS)

roads_match = roads_m[
    ["edge_id", "u", "v", "key", "geometry"]
].copy()

flood_match = flood_m[["geometry"]].copy()
flood_match = flood_match[
    flood_match.geometry.notna()
    & ~flood_match.geometry.is_empty
].reset_index(drop=True)

flood_match["_flood_id"] = range(len(flood_match))

print("Matching flood locations to nearest roads...")
matches = gpd.sjoin_nearest(
    flood_match,
    roads_match,
    how="left",
    max_distance=MATCH_DISTANCE,
    distance_col="distance_m"
)

# Count each source flood record at most once.
matches = (
    matches.dropna(subset=["edge_id"])
    .sort_values("distance_m")
    .drop_duplicates(subset="_flood_id")
)

matches["edge_id"] = matches["edge_id"].astype("int64")

counts = (
    matches.groupby(["u", "v", "key", "edge_id"])
    .size()
    .reset_index(name="nearby_flood_locations")
)

result = roads[["edge_id", "u", "v", "key"]].merge(
    counts,
    on=["edge_id", "u", "v", "key"],
    how="left",
    validate="one_to_one"
)

result["nearby_flood_locations"] = (
    result["nearby_flood_locations"].fillna(0).astype("int64")
)

result["flood_data_available"] = (
    result["nearby_flood_locations"] > 0
).astype("int64")

result.to_csv(OUTPUT_FILE, index=False)

print("\nFLOOD MATCHING REPORT")
print("Road edges:", len(roads))
print("Input flood records:", len(flood_match))
print("Matched flood records:", len(matches))
print("Edges near recorded flood locations:",
      int((result["nearby_flood_locations"] > 0).sum()))
print("Saved:", OUTPUT_FILE)
print("\nOutput columns:", result.columns.tolist())
