"""A2: multi-criteria, trip-round search on the shared time-dependent network.

The implementation scans the common connection adjacency while keeping Pareto
bags by stop/node and boarded-trip round. It is exact on the supplied transition
model; its Python scan is a research prototype rather than optimized RAPTOR.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import time

from pumo.model import Journey, Query, RouteResult
from pumo.objectives import OBJECTIVES
from .base import Router, edge_arrival, make_legs, query_start, step_values
from .label_setting import _dominates_label, _dominates_or_equal_label


@dataclass
class _RoundLabel:
    node: int
    arrival: int
    values: dict[str,float]
    boardings: int
    last_trip: str|None
    on_trip: str|None
    parent: int
    edge: object|None
    alive: bool=True


class McRaptorRouter(Router):
    name="A2"

    def route(self,q:Query)->RouteResult:
        if self.network is None: raise ValueError("router has no PreparedNetwork")
        started=time.perf_counter(); net=self.network
        source,departure=query_start(q,net); target=net.nearest_node(q.destination)
        max_boardings=q.max_transfers+1
        initial={name:0.0 for name in OBJECTIVES}; initial["T"]=float(departure)
        labels=[_RoundLabel(source,departure,initial,0,None,None,-1,None)]
        bags={(source,0,None,None):[0]}; queue=[(departure,0)]
        goals=[]; generated=expanded=0; timed_out=cap_hit=False; highest_round=0
        while queue:
            if time.perf_counter()-started>=q.timeout_s: timed_out=True; break
            _,index=heapq.heappop(queue); label=labels[index]
            if not label.alive: continue
            expanded+=1
            if label.node==target:
                if not any(_dominates_or_equal_label(labels[g],label,q.objectives) for g in goals if labels[g].alive):
                    goals=[g for g in goals if labels[g].alive and not _dominates_label(label,labels[g],q.objectives)]
                    goals.append(index)
                continue
            for edge in net.transitions_from(label.node,label.arrival,label.on_trip,disruptions=q.disruptions):
                arrival=edge_arrival(edge,label.arrival,on_trip=label.on_trip)
                if arrival is None or arrival<=label.arrival: continue
                new_boarding=edge.mode!="WALK" and edge.trip_id!=label.on_trip
                boardings=label.boardings+int(new_boarding)
                if boardings>max_boardings: continue
                trip=edge.trip_id if edge.mode!="WALK" else None
                last_trip=edge.trip_id if trip else label.last_trip
                values=step_values(label.values,edge,label.arrival,arrival,previous_trip=label.last_trip,
                                   on_trip=label.on_trip,network=net)
                proposed=_RoundLabel(edge.to_node,arrival,values,boardings,last_trip,trip,index,edge)
                key=(edge.to_node,boardings,last_trip,trip)
                if any(_dominates_or_equal_label(labels[j],proposed,q.objectives) for j in bags.get(key,()) if labels[j].alive): continue
                for j in bags.get(key,()):
                    if labels[j].alive and _dominates_label(proposed,labels[j],q.objectives): labels[j].alive=False
                new_idx=len(labels); labels.append(proposed); bags.setdefault(key,[]).append(new_idx)
                generated+=1; highest_round=max(highest_round,boardings)
                if generated>=self.label_cap: cap_hit=True; break
                heapq.heappush(queue,(arrival,new_idx))
            if cap_hit: break
        journeys=[]
        for goal in goals:
            transitions=[]; cursor=goal
            while labels[cursor].parent>=0:
                parent=labels[cursor].parent; edge=labels[cursor].edge
                contribution={k:labels[cursor].values.get(k,0)-labels[parent].values.get(k,0) for k in OBJECTIVES if k!="T"}
                contribution["T"]=labels[cursor].arrival-labels[parent].arrival
                transitions.append((edge,labels[cursor].arrival,contribution)); cursor=parent
            transitions.reverse(); totals=dict(labels[goal].values); totals["T"]=labels[goal].arrival-departure
            journeys.append(Journey(make_legs(transitions,departure),totals))
        return RouteResult(journeys,{"runtime_s":time.perf_counter()-started,"labels_generated":generated,
            "labels_expanded":expanded,"rounds":highest_round,"timed_out":timed_out,
            "label_cap_hit":cap_hit,"exact":not timed_out and not cap_hit,
            "implementation_note":"connection adjacency with round-indexed Pareto bags"})
