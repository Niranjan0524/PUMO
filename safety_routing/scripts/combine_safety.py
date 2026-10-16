import os
import pandas as pd

ROADS_FILE = "processed_data/road_edges.gpkg"
LIGHTS_FILE = "processed_data/streetlight_edge_safety.csv"
FLOOD_FILE = "processed_data/flood_edge_safety.csv"
OUTPUT_FILE = "processed_data/safety_edge_dataset_v2.csv"

KEYS = ["u", "v", "key"]

os.makedirs("processed_data", exist_ok=True)

print("Loading datasets...")

# GeoPandas is required to read the road GeoPackage.
import geopandas as gpd

roads = gpd.read_file(ROADS_FILE).reset_index(drop=True)
roads["edge_id"] = range(len(roads))

for col in KEYS:
    roads[col] = pd.to_numeric(roads[col], errors="raise").astype("int64")

lights = pd.read_csv(LIGHTS_FILE)
flood = pd.read_csv(FLOOD_FILE)

for name, frame in [("Streetlights", lights), ("Flood", flood)]:
    missing = [c for c in KEYS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} file lacks {missing}. "
            "Run the updated matching script first."
        )

    for col in KEYS:
        frame[col] = pd.to_numeric(
            frame[col], errors="raise"
        ).astype("int64")

if lights.duplicated(KEYS).any():
    raise ValueError("Duplicate road keys found in streetlight data.")

if flood.duplicated(KEYS).any():
    raise ValueError("Duplicate road keys found in flood data.")

road_keys = roads[KEYS]
if road_keys.duplicated(KEYS).any():
    raise ValueError("Road network has duplicate (u, v, key) keys.")

# Start from the canonical road list and merge using original IDs.
result = roads[["edge_id"] + KEYS].merge(
    lights[KEYS + [
        "streetlight_count",
        "streetlight_data_available"
    ]],
    on=KEYS,
    how="left",
    validate="one_to_one",
    indicator="_light_merge"
)

result = result.merge(
    flood[KEYS + [
        "nearby_flood_locations",
        "flood_data_available"
    ]],
    on=KEYS,
    how="left",
    validate="one_to_one",
    indicator="_flood_merge"
)

if len(result) != len(roads):
    raise ValueError("Merged row count does not match road network.")

if result.duplicated(KEYS).any():
    raise ValueError("Duplicate road keys appeared after merging.")

# Missing join matches are not automatically proof that no hazard exists.
# The matching scripts should contain one row per road edge.
if (result["_light_merge"] != "both").any():
    raise ValueError("Some road edges did not match the streetlight table.")

if (result["_flood_merge"] != "both").any():
    raise ValueError("Some road edges did not match the flood table.")

result = result.drop(columns=["_light_merge", "_flood_merge"])

result["streetlight_count"] = (
    result["streetlight_count"].astype("int64")
)
result["streetlight_data_available"] = (
    result["streetlight_data_available"].astype("int64")
)
result["nearby_flood_locations"] = (
    result["nearby_flood_locations"].astype("int64")
)
result["flood_data_available"] = (
    result["flood_data_available"].astype("int64")
)

result.to_csv(OUTPUT_FILE, index=False)

print("\nSAFETY DATASET CREATED")
print("Rows:", len(result))
print("Unique road keys:", result[KEYS].drop_duplicates().shape[0])
print("Edges with matched streetlights:",
      int((result["streetlight_count"] > 0).sum()))
print("Edges near recorded flood locations:",
      int((result["nearby_flood_locations"] > 0).sum()))
print("Saved:", OUTPUT_FILE)
print("Columns:", result.columns.tolist())
