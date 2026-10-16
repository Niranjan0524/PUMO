#!/usr/bin/env python3
"""Run a small route-aware disruption and mid-trip rerouting experiment."""
import argparse,json,sys,time
from pathlib import Path
import pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from pumo.algos.label_setting import LabelSettingRouter
from pumo.algos.mcraptor import McRaptorRouter
from pumo.algos.weight_sweep import WeightSweepRouter
from pumo.algos.weighted_astar import WeightedAStarRouter
from pumo.bench.disruptions import choose_disruption,shared_segment_ratio
from pumo.bench.metrics import result_metrics
from pumo.model import Query
from pumo.network import PreparedNetwork

ap=argparse.ArgumentParser(); ap.add_argument("scenario",choices=["core","mid"]); ap.add_argument("--query-limit",type=int,default=10); args=ap.parse_args()
scenario=args.scenario; cfg=yaml.safe_load((ROOT/"config"/"benchmark.yaml").read_text(encoding="utf-8")); scales=cfg["scales"]
qpath=ROOT/"derived_data"/"benchmark"/f"queries_{scenario}.parquet"
if not qpath.exists(): raise SystemExit(f"Generate queries first: python scripts/04_generate_queries.py {scenario}")
queries=pd.read_parquet(qpath).head(args.query_limit)
radius=cfg["scenarios"]["core_radius_km"] if scenario=="core" else cfg["scenarios"]["mid_radius_km"]
net=PreparedNetwork.from_directory(ROOT/"derived_data"/"processed",scenario=scenario,radius_km=radius)
profiles=pd.read_csv(ROOT/"config"/"profiles.csv"); balanced={k:float(profiles.iloc[-1][col]) for k,col in zip(("T","C","X","W","E","S"),profiles.columns[1:])}
rows=[]
for i,qrow in enumerate(queries.itertuples(index=False)):
    query=Query((float(qrow.origin_lat),float(qrow.origin_lon)),(float(qrow.destination_lat),float(qrow.destination_lon)),
        int(qrow.departure_time),("T","C","X","W","E","S"),balanced,timeout_s=float(cfg["timeout_s"]))
    baseline=WeightedAStarRouter(net,scales=scales,label_cap=int(cfg["label_cap"]),
        max_speed_cap_mps=float(cfg.get("max_speed_cap_mps",40.0))).route(query)
    if not baseline.journeys: continue
    rng_seed=int(cfg["seed"])+i
    activation=query.departure_time+int(pd.Series([900,1200,1500]).sample(n=1,random_state=rng_seed).iloc[0])
    event,state,suffix=choose_disruption(baseline.journeys[0],activation,seed=rng_seed)
    if event is None: continue
    reroute_q=Query(query.origin,query.destination,query.departure_time,query.objectives,query.profile_weights,
        (event,),state,query.max_transfers,query.timeout_s)
    outputs={}; stats={}; durations={}
    algorithms=["A1","A2","B0","W"] if scenario=="core" else ["A2","B0","W"]
    for name in algorithms:
        start=time.perf_counter()
        if name=="A1": result=LabelSettingRouter(net,label_cap=int(cfg["label_cap"])).route(reroute_q)
        elif name=="A2": result=McRaptorRouter(net,label_cap=int(cfg["label_cap"])).route(reroute_q)
        elif name=="W": result=WeightSweepRouter(net,scales=scales,sweep_size=int(cfg["sweep_size"]),seed=rng_seed,
            max_speed_cap_mps=float(cfg.get("max_speed_cap_mps",40.0))).route(reroute_q)
        else: result=WeightedAStarRouter(net,scales=scales,label_cap=int(cfg["label_cap"]),
            max_speed_cap_mps=float(cfg.get("max_speed_cap_mps",40.0))).route(reroute_q)
        durations[name]=time.perf_counter()-start; outputs[name]=result.journeys; stats[name]=result.stats
    if stats.get("A1",{}).get("exact"): reference=outputs["A1"]; ref_type="EXACT_A1_POST_DISRUPTION"
    else:
        from pumo.pareto import pareto_filter
        reference=pareto_filter([j for name in ("A2","W","B0") for j in outputs.get(name,[])],
            lambda j:tuple(j.vector[k] for k in query.objectives)); ref_type="BEST_KNOWN_POST_DISRUPTION"
    before=[s for leg in suffix for s in leg.segment_ids]
    for name in algorithms:
        selected=outputs[name][0] if outputs[name] else None
        after=[s for leg in selected.legs for s in leg.segment_ids] if selected else []
        quality=result_metrics(outputs[name],reference,query.objectives,global_scales=scales,
            profiles=[balanced],seed=rng_seed,mc_samples=int(cfg["monte_carlo_samples"]))
        rows.append({"query_id":qrow.query_id,"scenario":scenario,"algorithm":name,"event_id":event.event_id,
            "event_type":event.event_type,"activation_time":event.start_time,"reference_type":ref_type,
            "reroute_latency_s":durations[name],"journeys_returned":len(outputs[name]),
            "route_changed":bool(selected and after!=before),"shared_segment_ratio":shared_segment_ratio(
                [leg for leg in suffix],selected.legs if selected else []),
            "post_recall":quality.get("recall"),"post_hypervolume_ratio":quality.get("hypervolume_ratio"),
            "post_chebyshev_regret":quality.get("chebyshev_regret"),
            "labels_generated":stats[name].get("labels_generated",0),"timed_out":stats[name].get("timed_out",False),
            "label_cap_hit":stats[name].get("label_cap_hit",False)})
ROOT.joinpath("results").mkdir(exist_ok=True)
out=ROOT/"results"/f"disruptions_{scenario}.csv"; pd.DataFrame(rows).to_csv(out,index=False)
print(f"wrote {len(rows)} rerouting rows to {out}")
