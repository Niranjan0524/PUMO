"""Unit tests for RoadGraph and Dijkstra shortest paths (Algorithm A1)."""
from __future__ import annotations

import unittest
import os
import sys

# Add src to sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pumo_paratransit.network import RoadGraph, create_small_test_graph, create_bengaluru_synthetic_network
from pumo_paratransit.dijkstra import DijkstraRouter


class TestNetworkAndDijkstra(unittest.TestCase):

    def setUp(self):
        self.graph = create_small_test_graph()
        self.router = DijkstraRouter(self.graph)

    def test_graph_properties(self):
        self.assertEqual(len(self.graph.nodes), 5)
        # Verify node addition and retrieval
        neighbors_a = self.graph.get_neighbors("A")
        dest_nodes = [e.to_node for e in neighbors_a]
        self.assertIn("B", dest_nodes)
        self.assertIn("C", dest_nodes)

    def test_dijkstra_analytical_shortest_path(self):
        """Validates Dijkstra against known shortest paths on small validation graph.

        Paths:
          A -> B is 3 km, 30 km/h = 360 s
          B -> C is 4 km, 30 km/h = 480 s
          A -> C via B is 840 s (vs direct 1200 s)
        """
        tt_ab, dist_ab, path_ab = self.router.find_shortest_path("A", "B", weight_by="time")
        self.assertAlmostEqual(tt_ab, 360.0, places=1)
        self.assertAlmostEqual(dist_ab, 3.0, places=1)
        self.assertEqual(path_ab, ["A", "B"])

        tt_ac, dist_ac, path_ac = self.router.find_shortest_path("A", "C", weight_by="time")
        self.assertAlmostEqual(tt_ac, 840.0, places=1)
        self.assertAlmostEqual(dist_ac, 7.0, places=1)
        self.assertEqual(path_ac, ["A", "B", "C"])

    def test_caching(self):
        """Verifies that subsequent calls hit the cache."""
        t1 = self.router.get_travel_time("A", "D")
        cache_key = ("A", "D", "time")
        self.assertIn(cache_key, self.router._cache)
        t2 = self.router.get_travel_time("A", "D")
        self.assertEqual(t1, t2)

    def test_bengaluru_network_connectivity(self):
        """Ensures the Bengaluru synthetic network is fully connected."""
        blr_graph = create_bengaluru_synthetic_network()
        blr_router = DijkstraRouter(blr_graph)
        nodes = list(blr_graph.nodes.keys())

        # Test reachability from Majestic (N01) to all other hubs
        for node in nodes:
            if node != "N01":
                tt = blr_router.get_travel_time("N01", node)
                self.assertLess(tt, float("inf"), f"Node {node} unreachable from N01")


if __name__ == "__main__":
    unittest.main()
