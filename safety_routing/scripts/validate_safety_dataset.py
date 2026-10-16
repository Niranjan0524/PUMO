import os
import pandas as pd

INPUT_FILE = "processed_data/safety_edge_dataset_v2.csv"
REPORT_FILE = "processed_data/safety_data_quality_report.txt"

KEYS = ["u", "v", "key"]

df = pd.read_csv(INPUT_FILE)

required = KEYS + [
    "edge_id",
    "streetlight_count",
    "streetlight_data_available",
    "nearby_flood_locations",
    "flood_data_available"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

lines = []

def add(message=""):
    lines.append(str(message))

add("PUMO SAFETY DATA QUALITY REPORT")
add("=" * 40)
add(f"Total road edges: {len(df)}")
add(f"Unique edge IDs: {df['edge_id'].nunique()}")
add(f"Duplicate edge IDs: {df['edge_id'].duplicated().sum()}")
add(f"Unique (u, v, key) combinations: {len(df[KEYS].drop_duplicates())}")
add(f"Duplicate road keys: {df.duplicated(KEYS).sum()}")
add(f"Missing values:\n{df[required].isna().sum().to_string()}")

add("\nStreetlight coverage")
add(f"Edges with matched streetlights: {(df['streetlight_count'] > 0).sum()}")
add(f"Edges with zero matched streetlights: {(df['streetlight_count'] == 0).sum()}")
add(f"Matched streetlight count total: {df['streetlight_count'].sum()}")

add("\nFlood-location coverage")
add(f"Edges near recorded flood locations: {(df['nearby_flood_locations'] > 0).sum()}")
add(f"Total matched flood-location counts: {df['nearby_flood_locations'].sum()}")

add("\nAttribute summary")
add(df[[
    "streetlight_count",
    "nearby_flood_locations"
]].describe().to_string())

# Verify binary availability indicators agree with counts.
light_flag_errors = (
    df["streetlight_data_available"]
    != (df["streetlight_count"] > 0).astype(int)
).sum()

flood_flag_errors = (
    df["flood_data_available"]
    != (df["nearby_flood_locations"] > 0).astype(int)
).sum()

add("\nConsistency checks")
add(f"Streetlight flag inconsistencies: {light_flag_errors}")
add(f"Flood flag inconsistencies: {flood_flag_errors}")

os.makedirs("processed_data", exist_ok=True)

report = "\n".join(lines)
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\nReport saved to: {REPORT_FILE}")

if df[KEYS].duplicated().any():
    raise ValueError("Validation failed: duplicate road keys.")

if df["edge_id"].duplicated().any():
    raise ValueError("Validation failed: duplicate edge IDs.")

if df[required].isna().any().any():
    raise ValueError("Validation failed: missing required values.")

if light_flag_errors or flood_flag_errors:
    raise ValueError("Validation failed: inconsistent availability flags.")

print("\nValidation completed successfully.")

