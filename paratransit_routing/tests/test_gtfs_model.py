"""Unit tests for Paper 1 — GTFS-like Shared Auto Network (Digital Matatu)."""
from __future__ import annotations
import os
import sys
import unittest

test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pumo_paratransit.gtfs_model import (
    AutoStop, SharedAutoRoute, SharedAutoNetwork,
    create_bengaluru_shared_auto_network,
)


class TestGTFSModel(unittest.TestCase):

    def test_stop_creation_and_retrieval(self):
        net = SharedAutoNetwork()
        stop = AutoStop("S01", "Majestic", 12.97, 77.57, road_node_id="N01", zone_type="terminus")
        net.add_stop(stop)
        self.assertIn("S01", net.stops)
        self.assertEqual(net.stops["S01"].road_node_id, "N01")

    def test_route_ordered_stop_ids(self):
        net = SharedAutoNetwork()
        for sid in ["S01", "S02", "S03"]:
            net.add_stop(AutoStop(sid, sid, road_node_id=f"N{sid[-2:]}"))
        route = SharedAutoRoute("R01", "Test Route")
        route.add_stop("S01", 0)
        route.add_stop("S02", 600)
        route.add_stop("S03", 1200)
        net.add_route(route)
        self.assertEqual(net.get_road_nodes_for_route("R01"), ["N01", "N02", "N03"])

    def test_validation_detects_missing_stop(self):
        net = SharedAutoNetwork()
        net.add_stop(AutoStop("S01", "Stop A", road_node_id="N01"))
        route = SharedAutoRoute("R01", "Broken Route")
        route.add_stop("S01", 0)
        route.add_stop("S99", 600)  # S99 does not exist
        net.add_route(route)
        errors = net.validate()
        self.assertTrue(any("S99" in e for e in errors))

    def test_validation_detects_missing_road_node(self):
        net = SharedAutoNetwork()
        net.add_stop(AutoStop("S01", "Stop A", road_node_id="N01"))
        net.add_stop(AutoStop("S02", "Stop B", road_node_id=None))  # Missing road node
        route = SharedAutoRoute("R01", "Route")
        route.add_stop("S01", 0)
        route.add_stop("S02", 600)
        net.add_route(route)
        errors = net.validate()
        self.assertTrue(any("road_node_id" in e for e in errors))

    def test_bengaluru_network_valid(self):
        net = create_bengaluru_shared_auto_network()
        errors = net.validate()
        self.assertEqual(errors, [], f"Validation errors: {errors}")
        self.assertGreaterEqual(len(net.routes), 5)
        self.assertGreaterEqual(len(net.stops), 15)

    def test_serialisation_roundtrip(self, tmp_path=None):
        import tempfile
        net = create_bengaluru_shared_auto_network()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        net.save_json(path)
        net2 = SharedAutoNetwork.load_json(path)
        self.assertEqual(len(net.stops), len(net2.stops))
        self.assertEqual(len(net.routes), len(net2.routes))
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
