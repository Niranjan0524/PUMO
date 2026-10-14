
import os
import geopandas as gpd
import pandas as pd

# --------------------------------------------------
# FILE PATHS AND SETTINGS
# --------------------------------------------------

ROADS_FILE = "processed_data/road_edges.gpkg"
LIGHTS_FILE = "processed_data/streetlights_deduplicated.gpkg"
OUTPUT_FILE = "processed_data/streetlight_edge_safety.csv"

MAX_DISTANCE = 30  # metres
PROJECTED_CRS = "EPSG:32643"

os.makedirs("processed_data", exist_ok=True)

# --------------------------------------------------
# LOAD ROAD NETWORK
# --------------------------------------------------

print("Loading road edges...")

roads = gpd.read_file(ROADS_FILE).reset_index(drop=True)

required = ["u", "v", "key", "geometry"]
missing = [col for col in required if col not in roads.columns]

if missing:
    raise ValueError(f"Missing road columns: {missing}")

if roads.crs is None:
    raise ValueError("Road network has no CRS defined.")

# Preserve the original road-file order for edge IDs.
roads["edge_id"] = range(len(roads))

for col in ["u", "v", "key"]:
    roads[col] = pd.to_numeric(roads[col], errors="raise").astype("int64")

roads_m = roads.to_crs(PROJECTED_CRS)

road_lookup = roads_m[
    ["edge_id", "u", "v", "key", "geometry"]
].copy()

# --------------------------------------------------
# LOAD STREETLIGHTS
# --------------------------------------------------

print("Loading streetlights...")

lights = gpd.read_file(LIGHTS_FILE)

if lights.crs is None:
    raise ValueError("Streetlights have no CRS defined.")

lights_m = lights.to_crs(PROJECTED_CRS)

lights_m = lights_m[
    lights_m.geometry.notna() & ~lights_m.geometry.is_empty
].copy()

# Give every input streetlight a stable ID for this run.
lights_m = lights_m.reset_index(drop=True)
lights_m["_light_id"] = range(len(lights_m))

print("Road edges:", len(road_lookup))
print("Input streetlights:", len(lights_m))
print("Matching each streetlight to its nearest road...")

# --------------------------------------------------
# MATCH EACH STREETLIGHT TO ITS NEAREST ROAD
# --------------------------------------------------

matches = gpd.sjoin_nearest(
    lights_m[["_light_id", "geometry"]],
    road_lookup,
    how="left",
    max_distance=MAX_DISTANCE,
    distance_col="distance_m"
)

# A streetlight can have multiple nearest-road matches
# if two roads are equally close. Keep the closest one.
matches = matches.dropna(
    subset=["edge_id", "distance_m"]
).copy()

matches = matches.sort_values(
    ["_light_id", "distance_m", "edge_id"]
)

matches = matches.drop_duplicates(
    subset="_light_id",
    keep="first"
)

matches["edge_id"] = matches["edge_id"].astype("int64")

# --------------------------------------------------
# COUNT UNIQUE MATCHED STREETLIGHTS PER ROAD EDGE
# --------------------------------------------------

counts = (
    matches.groupby("edge_id")["_light_id"]
    .nunique()
    .rename("streetlight_count")
    .reset_index()
)

result = road_lookup[
    ["edge_id", "u", "v", "key"]
].merge(
    counts,
    on="edge_id",
    how="left",
    validate="one_to_one"
)

result["streetlight_count"] = (
    result["streetlight_count"]
    .fillna(0)
    .astype("int64")
)

result["streetlight_data_available"] = (
    result["streetlight_count"] > 0
).astype("int64")

# --------------------------------------------------
# SAVE EDGE-LEVEL DATASET
# --------------------------------------------------

result.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# REPORT
# --------------------------------------------------

matched_lights = matches["_light_id"].nunique()
unmatched_lights = len(lights_m) - matched_lights
edges_with_lights = int(
    (result["streetlight_count"] > 0).sum()
)

print("\nSTREETLIGHT MATCHING REPORT")
print("---------------------------")
print("Road edges:", len(road_lookup))
print("Input streetlights:", len(lights_m))
print("Unique matched streetlights:", matched_lights)
print("Unmatched streetlights:", unmatched_lights)
print("Edges with matched streetlights:", edges_with_lights)
print("Edges without matched streetlights:",
      len(result) - edges_with_lights)
print("Maximum matching distance:", MAX_DISTANCE, "metres")
print("Saved:", OUTPUT_FILE)

print("\nStreetlight count summary:")
print(result["streetlight_count"].describe())

if not matches.empty:
    print("\nNearest-match distance summary (metres):")
    print(matches["distance_m"].describe())

    print("\nRoad edges with the highest streetlight counts:")
    print(
        result.nlargest(10, "streetlight_count")[
            ["edge_id", "u", "v", "key", "streetlight_count"]
        ].to_string(index=False)
    )
else:
    print("\nWARNING: No streetlights matched within the distance limit.")

print("\nMatching completed.")

