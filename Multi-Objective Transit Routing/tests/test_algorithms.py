from pumo.algos.label_setting import LabelSettingRouter
from pumo.algos.mcraptor import McRaptorRouter
from pumo.algos.weight_sweep import WeightSweepRouter
from pumo.algos.weighted_astar import WeightedAStarRouter
from pumo.algos.weighted_astar import time_lower_bound
from pumo.model import Disruption, JourneyState, Query, RouteResult
from pumo.network import PreparedNetwork, Transition
from pumo.algos.base import edge_arrival
from pumo.objectives import OBJECTIVES


def unsupported_toy_network():
    # Three direct services have (T,C) = (1,8), (4,6), (8,1).
    # (4,6) is Pareto-optimal but unsupported by any non-negative weighted sum.
    coords = {0: (12.90000,77.50000), 1:(12.90001,77.50001),
              2:(12.90002,77.50002), 3:(12.90003,77.50003)}
    net=PreparedNetwork(coords,boarding_buffer_s=0,fare_per_boarding={"BUS":0},coordinates_are_wgs84=True)
    for i,(duration,cost) in enumerate(((1,8),(4,6),(8,1))):
        net.add(Transition(0,3,"BUS",f"trip-{i}",distance_m=4.6,cost=cost,
            trip_id=f"trip-{i}",pattern_id=i,departure_time=0,arrival_time=duration))
    return net.finalize()


def query(weights=None):
    return Query((12.9,77.5),(12.90003,77.50003),0, ("T","C"), weights, timeout_s=5)


def test_a1_and_a2_reproduce_hand_computed_pareto_set():
    net=unsupported_toy_network()
    expected={(1.0,8.0),(4.0,6.0),(8.0,1.0)}
    for router in (LabelSettingRouter(net),McRaptorRouter(net)):
        result=router.route(query())
        assert result.stats["exact"]
        assert {(j.vector["T"],j.vector["C"]) for j in result.journeys}==expected


def test_a1_partial_result_flags_label_cap_and_keeps_discovered_target():
    result=LabelSettingRouter(unsupported_toy_network(),label_cap=1).route(query())
    assert result.stats["label_cap_hit"] and not result.stats["exact"]
    assert len(result.journeys)==1
    assert result.journeys[0].vector["T"]==1


def test_a1_counts_trip_changes_as_transfers_and_respects_limit():
    net=PreparedNetwork({0:(12.9,77.5),1:(12.90001,77.50001),2:(12.90002,77.50002)},
                        boarding_buffer_s=0, fare_per_boarding={"BUS":0}, coordinates_are_wgs84=True)
    net.add(Transition(0,1,"BUS","A:0>1",trip_id="A",pattern_id="A",departure_time=0,arrival_time=10))
    net.add(Transition(1,2,"BUS","B:1>2",trip_id="B",pattern_id="B",departure_time=20,arrival_time=30))
    net.finalize()
    q=Query((12.9,77.5),(12.90002,77.50002),0,("T","X"),timeout_s=5)
    result=LabelSettingRouter(net).route(q)
    assert result.stats["exact"]
    journey=result.journeys[0]
    assert journey.vector["X"]==1
    assert journey.vector["T"]==30  # includes the 10-second transfer wait
    assert [leg.dep_time for leg in journey.legs]==[0,20]
    limited=LabelSettingRouter(net).route(Query(q.origin,q.destination,0,("T","X"),
                                                max_transfers=0,timeout_s=5))
    assert limited.stats["exact"] and limited.journeys==[]


def test_a1_timeout_is_reported_as_inexact():
    q=Query((12.9,77.5),(12.90003,77.50003),0,("T","C"),timeout_s=1e-12)
    result=LabelSettingRouter(unsupported_toy_network()).route(q)
    assert result.stats["timed_out"] and not result.stats["exact"]


def test_a2_uses_walking_access_and_egress_paths():
    net=PreparedNetwork({0:(12.9,77.5),1:(12.90001,77.50001),
                         2:(12.90002,77.50002),3:(12.90003,77.50003)},
                        boarding_buffer_s=0,coordinates_are_wgs84=True)
    net.add(Transition(0,1,"WALK","access",walk_time_s=5,distance_m=5))
    net.add(Transition(1,2,"BUS","bus",trip_id="bus",pattern_id="bus",
                       departure_time=10,arrival_time=20))
    net.add(Transition(2,3,"WALK","egress",walk_time_s=4,distance_m=4))
    net.stop_nodes={10:1,20:2}; net.node_stops={1:[10],2:[20]}
    net.finalize()
    q=Query((12.9,77.5),(12.90003,77.50003),0,("T","C","X","W"),timeout_s=5)
    result=McRaptorRouter(net).route(q)
    assert result.stats["exact"] and result.stats["rounds"]==1
    journey=result.journeys[0]
    assert [leg.mode for leg in journey.legs]==["WALK","BUS","WALK"]
    assert journey.vector["T"]==24
    assert journey.vector["W"]==9


def test_a2_relaxes_transfer_footpaths_and_matches_a1():
    net=PreparedNetwork({0:(12.9,77.5),1:(12.90001,77.50001),
                         2:(12.90002,77.50002),3:(12.90003,77.50003)},
                        boarding_buffer_s=0,fare_per_boarding={"BUS":0},coordinates_are_wgs84=True)
    net.add(Transition(0,1,"BUS","A:0>1",trip_id="A",pattern_id="A",
                       departure_time=0,arrival_time=10))
    footpath=Transition(1,2,"WALK","footpath:1>2",walk_time_s=5,distance_m=5,
                        segment_ids=("walk-a","walk-b"))
    net.add(footpath); net.add_footpath(footpath)
    net.add(Transition(2,3,"BUS","B:2>3",trip_id="B",pattern_id="B",
                       departure_time=20,arrival_time=30))
    net.stop_nodes={10:0,11:1,12:2,13:3}; net.node_stops={0:[10],1:[11],2:[12],3:[13]}
    net.finalize()
    q=Query((12.9,77.5),(12.90003,77.50003),0,("T","C","X","W"),timeout_s=5)
    a2=McRaptorRouter(net).route(q)
    a1=LabelSettingRouter(net).route(q)
    assert a2.stats["exact"] and a2.stats["rounds"]==2
    assert a2.journeys[0].vector["X"]==1
    assert a2.journeys[0].vector["T"]==30
    assert a2.journeys[0].legs[1].segment_ids==["walk-a","walk-b"]
    assert {(j.vector["T"],j.vector["X"],j.vector["W"]) for j in a2.journeys}=={
        (j.vector["T"],j.vector["X"],j.vector["W"]) for j in a1.journeys}
    limited=McRaptorRouter(net).route(Query(q.origin,q.destination,0,q.objectives,
                                           max_transfers=0,timeout_s=5))
    assert limited.stats["exact"] and limited.journeys==[]


def test_a2_adds_walking_only_journey():
    net=PreparedNetwork({0:(12.9,77.5),1:(12.90001,77.50001)},coordinates_are_wgs84=True)
    net.add(Transition(0,1,"WALK","walk",walk_time_s=5,distance_m=5))
    net.finalize()
    q=Query((12.9,77.5),(12.90001,77.50001),0,("T","W"),timeout_s=5)
    result=McRaptorRouter(net).route(q)
    assert result.stats["exact"]
    assert len(result.journeys)==1
    assert result.journeys[0].legs[0].segment_ids==["walk"]


def test_b0_selects_the_correct_weighted_sum_extremes():
    net=unsupported_toy_network()
    router=WeightedAStarRouter(net,scales={"T":1,"C":1})
    fast=router.route(query({"T":1,"C":0}))
    cheap=router.route(query({"T":0,"C":1}))
    assert fast.journeys[0].vector["T"]==1
    assert cheap.journeys[0].vector["C"]==1


def test_b0_keeps_early_expensive_labels_that_can_catch_a_service():
    net=PreparedNetwork({0:(12.9,77.5),1:(12.90001,77.50001),2:(12.90002,77.50002)},
                        boarding_buffer_s=0,coordinates_are_wgs84=True)
    # The earlier walk costs more W but catches the only useful bus. A single
    # lowest-score label at node 1 would discard it and return W=21 instead.
    net.add(Transition(0,1,"WALK","fast",walk_time_s=1,distance_m=10))
    net.add(Transition(0,1,"WALK","cheap",walk_time_s=5,distance_m=1))
    net.add(Transition(1,2,"BUS","bus",trip_id="bus",pattern_id="bus",
                       departure_time=2,arrival_time=3))
    net.add(Transition(1,2,"WALK","detour",walk_time_s=100,distance_m=20))
    net.finalize()
    q=Query((12.9,77.5),(12.90002,77.50002),0,("W",),{"W":1},timeout_s=5)
    result=WeightedAStarRouter(net,scales={"W":1}).route(q)
    assert result.stats["exact"]
    assert result.journeys[0].vector["W"]==10
    assert result.journeys[0].legs[0].segment_ids==["fast"]


def test_b0_preserves_a_discovered_target_when_label_cap_hits():
    result=WeightedAStarRouter(unsupported_toy_network(),scales={"T":1,"C":1},
                               label_cap=1).route(query({"T":1,"C":0}))
    assert result.stats["label_cap_hit"] and not result.stats["exact"]
    assert len(result.journeys)==1 and result.journeys[0].vector["T"]==1


def test_weight_sweep_is_nondominated_and_misses_unsupported_point():
    net=unsupported_toy_network()
    router=WeightSweepRouter(net,scales={"T":1,"C":1},sweep_size=30,seed=7,profiles=[])
    result=router.route(query())
    pairs={(j.vector["T"],j.vector["C"]) for j in result.journeys}
    assert (4.0,6.0) not in pairs
    assert (1.0,8.0) in pairs and (8.0,1.0) in pairs
    for a in result.journeys:
        for b in result.journeys:
            if a is not b:
                assert not (a.vector["T"]<=b.vector["T"] and a.vector["C"]<=b.vector["C"] and
                           (a.vector["T"]<b.vector["T"] or a.vector["C"]<b.vector["C"]))
    assert result.stats["weight_vectors_run"]==30
    assert len(result.stats["per_weight_runtime_s"])==30
    assert result.stats["supported_front_only"]


def test_weight_sweep_shares_query_timeout_across_vectors(monkeypatch):
    class TimedOutRouter:
        def __init__(self,*args,**kwargs): pass
        def route(self,q):
            return RouteResult([],{"runtime_s":q.timeout_s,"timed_out":True,
                                   "label_cap_hit":False,"exact":False})
    monkeypatch.setattr("pumo.algos.weight_sweep.WeightedAStarRouter",TimedOutRouter)
    result=WeightSweepRouter(unsupported_toy_network(),scales={"T":1,"C":1},
                             sweep_size=30,profiles=[]).route(query())
    assert result.stats["weight_vectors_run"]==1
    assert result.stats["timed_out"] and not result.stats["exact"]


def test_weight_sweep_seeded_vectors_and_runtime_stats_are_reproducible():
    net=unsupported_toy_network()
    first=WeightSweepRouter(net,scales={"T":1,"C":1},sweep_size=12,seed=91,profiles=[]).route(query())
    second=WeightSweepRouter(net,scales={"T":1,"C":1},sweep_size=12,seed=91,profiles=[]).route(query())
    assert first.stats["weight_vectors"]==second.stats["weight_vectors"]
    assert first.stats["weight_vectors_run"]==12
    assert len(first.stats["per_weight_runtime_s"])==12
    assert first.stats["distinct_journeys"]>=len(first.journeys)


def test_weight_sweep_contains_units_profiles_and_seeded_simplex_samples():
    router=WeightSweepRouter(scales={key:1 for key in OBJECTIVES},sweep_size=50,seed=13)
    vectors=router.weight_vectors(OBJECTIVES)
    assert len(vectors)==50
    assert [tuple(v[key] for key in OBJECTIVES) for v in vectors[:6]]==[
        tuple(float(key==name) for key in OBJECTIVES) for name in OBJECTIVES]
    assert all(abs(sum(vector.values())-1.0)<1e-12 for vector in vectors[6:])
    again=WeightSweepRouter(scales={key:1 for key in OBJECTIVES},sweep_size=50,seed=13).weight_vectors(OBJECTIVES)
    assert vectors==again


def test_disruption_overlay_closes_edges_and_shifts_transit_trip():
    net=unsupported_toy_network()
    road=Transition(0,1,"WALK","1",walk_time_s=10,distance_m=10)
    net.add(road)
    closed=Disruption("d1","ROAD_CLOSURE",0,100,edge_ids=(1,),closure=True)
    assert road not in list(net.transitions_from(0,0,None,disruptions=(closed,)))
    trip=Transition(0,3,"BUS","trip-delay",trip_id="trip-delay",pattern_id="delay",
                    departure_time=10,arrival_time=20)
    net.add(trip)
    delayed=Disruption("d2","BUS_DELAY",10,100,trip_ids=("trip-delay",),time_penalty_s=300)
    shifted=[e for e in net.transitions_from(0,0,None,disruptions=(delayed,)) if e.trip_id=="trip-delay"]
    assert len(shifted)==1
    assert shifted[0].departure_time==310 and edge_arrival(shifted[0],0,on_trip=None)==320


def test_reroute_starts_at_current_node_and_time():
    net=unsupported_toy_network()
    net.add(Transition(0,3,"BUS","later",trip_id="later",pattern_id="later",
                       departure_time=60,arrival_time=70))
    q=Query((12.9,77.5),(12.90003,77.50003),0,("T","C"),{"T":1,"C":0},
            start_state=JourneyState("node:0",50),timeout_s=5)
    result=WeightedAStarRouter(net,scales={"T":1,"C":1}).route(q)
    assert result.journeys[0].legs[0].dep_time>=50
    assert result.journeys[0].vector["T"]>=0


def test_single_objective_agreement_and_seeded_determinism():
    net=unsupported_toy_network()
    q=Query((12.9,77.5),(12.90003,77.50003),0,("T",),timeout_s=5)
    results=[LabelSettingRouter(net).route(q),McRaptorRouter(net).route(q),
             WeightedAStarRouter(net,scales={"T":1}).route(Query(q.origin,q.destination,0,("T",),{"T":1},timeout_s=5)),
             WeightSweepRouter(net,scales={"T":1},sweep_size=12,seed=123,profiles=[]).route(q)]
    fastest=min(j.vector["T"] for j in results[0].journeys)
    assert all(min(j.vector["T"] for j in r.journeys)==fastest for r in results)
    w1=WeightSweepRouter(net,scales={"T":1},sweep_size=12,seed=123,profiles=[]).route(q)
    w2=WeightSweepRouter(net,scales={"T":1},sweep_size=12,seed=123,profiles=[]).route(q)
    assert [j.vector for j in w1.journeys]==[j.vector for j in w2.journeys]


def test_b0_heuristic_does_not_overestimate_toy_remaining_time():
    net=unsupported_toy_network()
    speed=max((e.distance_m/(e.arrival_time-e.departure_time)) for edges in net.transit.values() for e in edges)
    estimate=time_lower_bound(net,0,3,{"T":1},{"T":1},speed,("T",))
    exact=min(e.arrival_time-e.departure_time for e in net.transit[0])
    assert abs(net.straight_line_distance_m(0,3)-4.6)<0.2
    assert estimate<=exact+1e-9
    disabled=WeightedAStarRouter(net,scales={"T":1},max_speed_cap_mps=1.0).route(
        Query((12.9,77.5),(12.90003,77.50003),0,("T",),{"T":1},timeout_s=5))
    assert disabled.stats["exact"] and not disabled.stats["heuristic_enabled"]


def test_dropping_objective_keeps_fastest_arrival():
    net=unsupported_toy_network()
    full=LabelSettingRouter(net).route(query())
    time_only=LabelSettingRouter(net).route(Query(query().origin,query().destination,0,("T",),timeout_s=5))
    assert min(j.vector["T"] for j in full.journeys)==min(j.vector["T"] for j in time_only.journeys)


def test_seeded_tiny_networks_match_exhaustive_oracle():
    import numpy as np
    for seed in range(5):
        rng=np.random.default_rng(seed); alternatives=[]
        net=PreparedNetwork({0:(12.9,77.5),1:(12.90003,77.50003)},boarding_buffer_s=0,
                            fare_per_boarding={"BUS":0},coordinates_are_wgs84=True)
        for i in range(6):
            duration=int(rng.integers(1,25)); cost=int(rng.integers(1,25))
            alternatives.append((duration,cost))
            net.add(Transition(0,1,"BUS",f"{seed}-{i}",distance_m=4.6,cost=cost,
                trip_id=f"{seed}-{i}",pattern_id=i,departure_time=0,arrival_time=duration))
        net.finalize()
        oracle=[]
        for candidate in alternatives:
            if not any(a[0]<=candidate[0] and a[1]<=candidate[1] and a!=candidate for a in alternatives):
                if candidate not in oracle: oracle.append(candidate)
        q=Query((12.9,77.5),(12.90003,77.50003),0,("T","C"),timeout_s=5)
        for router in (LabelSettingRouter(net),McRaptorRouter(net)):
            actual={(j.vector["T"],j.vector["C"]) for j in router.route(q).journeys}
            assert actual==set(oracle)
        weights={"T":float(rng.random()),"C":float(rng.random())}
        scales={"T":1.0,"C":1.0}
        score=lambda pair:weights["T"]*pair[0]+weights["C"]*pair[1]
        optimum=min(score(pair) for pair in alternatives)
        b0=WeightedAStarRouter(net,scales=scales).route(Query(q.origin,q.destination,0,q.objectives,weights,timeout_s=5))
        assert b0.stats["exact"]
        assert abs(b0.stats["weighted_score"]-optimum)<1e-9
        w=WeightSweepRouter(net,scales=scales,sweep_size=20,seed=seed,profiles=[]).route(q)
        assert {(j.vector["T"],j.vector["C"]) for j in w.journeys} <= set(oracle)
