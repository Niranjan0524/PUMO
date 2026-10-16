from pumo.walking import WalkingGraph


def test_bounded_walking_keeps_short_risky_and_long_safe_options():
    # Two parallel routes: 0-1-3 is fast/risky; 0-2-3 is slower/safe.
    graph = WalkingGraph(4,
        from_node=[0, 1, 0, 2], to_node=[1, 3, 2, 3], edge_id=[10, 11, 20, 21],
        time_s=[1, 1, 2, 2], distance_m=[10, 10, 20, 20], risk=[50, 50, 0, 0])
    options = graph.bounded_options(0, [3], max_distance_m=50, max_time_s=10, limit=3)
    assert len(options) == 2
    assert {(o.walk_time_s, o.distance_m, o.risk_exposure) for o in options} == {
        (2, 20, 100), (4, 40, 0)
    }
    assert {o.edge_ids for o in options} == {(10, 11), (20, 21)}
