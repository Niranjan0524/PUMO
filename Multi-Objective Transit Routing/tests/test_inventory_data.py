from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inventory", ROOT / "scripts" / "00_inventory.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def test_gtfs_feed_classification():
    assert inventory.gtfs_summary(ROOT / "Data" / "bmtc")["classification"] == "BUS"
    assert inventory.gtfs_summary(ROOT / "Data" / "bmrcl")["classification"] == "METRO"


def test_inventory_recognizes_aggregate_accidents_and_flood_points():
    accidents = inventory.csv_summary(ROOT / "Data" / "Accident data.csv")
    assert not accidents["latitude_like"]
    assert not accidents["longitude_like"]
    flood = inventory.kml_summary(ROOT / "Data" / "floodZone data.kml")
    assert flood["feature_count"] == 200


def test_profile_weights_sum_to_one():
    profiles = pd.read_csv(ROOT / "config" / "profiles.csv")
    assert all(abs(row - 1.0) < 1e-9 for row in profiles.iloc[:, 1:].sum(axis=1))


def test_gtfs_times_keep_day_rollover():
    spec = importlib.util.spec_from_file_location("prepare_network", ROOT / "scripts" / "01_prepare_network.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.gtfs_time("24:30:00") == 88200
    assert module.gtfs_time("25:00:00") == 90000
