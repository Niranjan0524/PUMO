"""B0: weighted-sum time-dependent A* with an arrival/score label frontier."""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import time

from pumo.model import Journey, Query, RouteResult
from pumo.objectives import OBJECTIVES, weighted_score
from .base import Router, edge_arrival, make_legs, query_start, step_values


@dataclass
class _ScalarLabel:
    node: int
    arrival: int
    values: dict[str, float]
    last_trip: str | None
    on_trip: str | None
    parent: int
    edge: object | None
    score: float
    alive: bool = True


class WeightedAStarRouter(Router):
    name = "B0"

    def __init__(self, network=None, *, scales=None, label_cap=2_000_000, max_speed_mps=None):
        super().__init__(network, label_cap=label_cap)
        self.scales = scales or {k: 1.0 for k in OBJECTIVES}
        self.max_speed_mps = max_speed_mps

    def route(self, q: Query) -> RouteResult:
        if self.network is None: raise ValueError("router has no PreparedNetwork")
        if not q.profile_weights: raise ValueError("B0 requires profile_weights")
        started=time.perf_counter(); net=self.network
        source,departure=query_start(q,net); target=net.nearest_node(q.destination)
        max_speed=self.max_speed_mps or _max_speed(net)
        weights={k:float(q.profile_weights.get(k,0.0)) for k in q.objectives}
        def score(v):
            values=dict(v); values["T"]=max(0.0,values.get("T",departure)-departure)
            return weighted_score(values,weights,self.scales,q.objectives)
        def heuristic(node):
            return time_lower_bound(net,node,target,weights,self.scales,max_speed,q.objectives)
        initial={name:0.0 for name in OBJECTIVES}; initial["T"]=float(departure)
        labels=[_ScalarLabel(source,departure,initial,None,None,-1,None,0.0)]
        bags={(source,None,None):[0]}; queue=[(heuristic(source),0.0,departure,0)]
        best_goal=None; generated=expanded=0; timed_out=cap_hit=False
        while queue:
            if time.perf_counter()-started >= q.timeout_s: timed_out=True; break
            f,g,_,i=heapq.heappop(queue); label=labels[i]
            if not label.alive: continue
            if best_goal is not None and f >= labels[best_goal].score-1e-12: break
            expanded+=1
            if label.node==target:
                if best_goal is None or label.score<labels[best_goal].score: best_goal=i
                continue
            for edge in net.transitions_from(label.node,label.arrival,label.on_trip,disruptions=q.disruptions):
                arrival=edge_arrival(edge,label.arrival,on_trip=label.on_trip)
                if arrival is None or arrival<=label.arrival: continue
                trip=edge.trip_id if edge.mode!="WALK" else None
                last_trip=edge.trip_id if trip else label.last_trip
                values=step_values(label.values,edge,label.arrival,arrival,previous_trip=label.last_trip,
                                   on_trip=label.on_trip,network=net)
                if values.get("X",0.0)>q.max_transfers: continue
                candidate_score=score(values)
                key=(edge.to_node,last_trip,trip)
                current=bags.get(key,[])
                if any(labels[j].alive and labels[j].arrival<=arrival and labels[j].score<=candidate_score+1e-12 for j in current): continue
                for j in current:
                    if labels[j].alive and arrival<=labels[j].arrival and candidate_score<=labels[j].score+1e-12: labels[j].alive=False
                ni=len(labels); labels.append(_ScalarLabel(edge.to_node,arrival,values,last_trip,trip,i,edge,candidate_score))
                bags.setdefault(key,[]).append(ni); generated+=1
                if generated>=self.label_cap: cap_hit=True; break
                heapq.heappush(queue,(candidate_score+heuristic(edge.to_node),candidate_score,arrival,ni))
            if cap_hit: break
        journeys=[]
        if best_goal is not None:
            transitions=[]; cursor=best_goal
            while labels[cursor].parent>=0:
                parent=labels[cursor].parent; edge=labels[cursor].edge
                contribution={k:labels[cursor].values.get(k,0)-labels[parent].values.get(k,0) for k in OBJECTIVES if k!="T"}
                contribution["T"]=labels[cursor].arrival-labels[parent].arrival
                transitions.append((edge,labels[cursor].arrival,contribution)); cursor=parent
            transitions.reverse(); totals=dict(labels[best_goal].values); totals["T"]=labels[best_goal].arrival-departure
            journeys=[Journey(make_legs(transitions,departure),totals)]
        return RouteResult(journeys,{"runtime_s":time.perf_counter()-started,"labels_generated":generated,
            "labels_expanded":expanded,"timed_out":timed_out,"label_cap_hit":cap_hit,
            "exact":not timed_out and not cap_hit,"weighted_score":labels[best_goal].score if best_goal is not None else None,
            "max_speed_mps":max_speed})


def _max_speed(net):
    values=[]
    for adjacency in (net.walk,net.transit):
        for edges in adjacency.values():
            for edge in edges:
                duration=(edge.walk_time_s if edge.mode=="WALK" else (edge.arrival_time-edge.departure_time))
                if duration and duration>0 and edge.distance_m>0: values.append(edge.distance_m/duration)
    return max(values,default=1.33)


def time_lower_bound(net,node,target,weights,scales,max_speed,objectives=OBJECTIVES):
    if "T" not in objectives or float(weights.get("T",0.0))==0: return 0.0
    a=net.coords.get(int(node),(0,0)); b=net.coords.get(int(target),(0,0))
    distance=max(0.0,((a[0]-b[0])**2+(a[1]-b[1])**2)**.5)
    return float(weights["T"])*distance/max(float(max_speed),1e-12)/float(scales["T"])
