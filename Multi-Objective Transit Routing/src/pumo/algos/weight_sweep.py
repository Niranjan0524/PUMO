"""W: deterministic weighted-sum sweep followed by shared Pareto filtering."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from pumo.model import Query, RouteResult
from pumo.objectives import OBJECTIVES
from pumo.pareto import pareto_filter
from .base import Router
from .weighted_astar import WeightedAStarRouter


class WeightSweepRouter(Router):
    name="W"

    def __init__(self, network=None, *, scales=None, sweep_size=50, seed=20261016,
                 profiles=None, label_cap=2_000_000):
        super().__init__(network,label_cap=label_cap)
        self.scales=scales or {k:1.0 for k in OBJECTIVES}; self.sweep_size=int(sweep_size); self.seed=int(seed)
        if profiles is None:
            p=__import__("pathlib").Path(__file__).resolve().parents[3]/"config"/"profiles.csv"
            if p.exists():
                frame=pd.read_csv(p).set_index("profile_id")
                profiles=[{k:float(frame.loc[i, col]) for k,col in zip(OBJECTIVES,frame.columns)} for i in frame.index]
            else: profiles=[]
        self.profiles=profiles

    def weight_vectors(self, objectives):
        names=tuple(objectives); vectors=[]
        for name in names: vectors.append({k:float(k==name) for k in names})
        for profile in self.profiles:
            restricted={k:max(0.0,float(profile.get(k,0.0))) for k in names}
            total=sum(restricted.values())
            if total: vectors.append({k:v/total for k,v in restricted.items()})
        unique=[]; seen=set()
        for vector in vectors:
            key=tuple(round(vector[k],12) for k in names)
            if key not in seen: seen.add(key); unique.append(vector)
        rng=np.random.default_rng(self.seed)
        remaining=max(0,self.sweep_size-len(unique))
        for sample in rng.dirichlet(np.ones(len(names)),size=remaining):
            unique.append(dict(zip(names,map(float,sample))))
        return unique

    def route(self,q:Query)->RouteResult:
        if self.network is None: raise ValueError("router has no PreparedNetwork")
        vectors=self.weight_vectors(q.objectives); started=time.perf_counter(); journeys=[]; runs=[]
        router=WeightedAStarRouter(self.network,scales=self.scales,label_cap=self.label_cap)
        for weights in vectors:
            result=router.route(Query(q.origin,q.destination,q.departure_time,q.objectives,weights,
                q.disruptions,q.start_state,q.max_transfers,q.timeout_s))
            runs.append(result.stats)
            journeys.extend(result.journeys)
        by_path={}
        for journey in journeys:
            key=tuple(s for leg in journey.legs for s in leg.segment_ids)
            by_path.setdefault(key,journey)
        selected=tuple(q.objectives)
        result=pareto_filter(by_path.values(),lambda j:tuple(j.vector.get(k,0.0) for k in selected))
        runtime=time.perf_counter()-started
        return RouteResult(result,{"runtime_s":runtime,"total_runtime_s":runtime,
            "weight_vectors_run":len(vectors),"distinct_journeys":len(by_path),
            "mean_runtime_per_weight_s":runtime/max(1,len(vectors)),
            "labels_generated":sum(r.get("labels_generated",0) for r in runs),
            "labels_expanded":sum(r.get("labels_expanded",0) for r in runs),
            "timed_out":any(r.get("timed_out") for r in runs),"label_cap_hit":any(r.get("label_cap_hit") for r in runs),
            "exact":all(r.get("exact",False) for r in runs),"supported_front_only":True})
