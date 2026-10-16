#!/usr/bin/env python3
"""Run the registered comparison for one scenario. Requires frozen Core scales."""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys, time, tracemalloc
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pumo.algos.label_setting import LabelSettingRouter
from pumo.algos.mcraptor import McRaptorRouter
from pumo.algos.weight_sweep import WeightSweepRouter
from pumo.algos.weighted_astar import WeightedAStarRouter
from pumo.bench.metrics import result_metrics
from pumo.model import Query
from pumo.network import PreparedNetwork
from pumo.objectives import OBJECTIVES

OBJECTIVE_SETS=(("T","C"),("T","C","X"),("T","C","X","W"),
                ("T","C","X","W","E"),OBJECTIVES)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("scenario",choices=["core","mid","city"])
    ap.add_argument("--query-limit",type=int,default=None)
    ap.add_argument("--algorithms",nargs="+",choices=["A1","A2","B0","W"],default=["A1","A2","B0","W"])
    ap.add_argument("--repetitions",type=int,default=None)
    args=ap.parse_args(); scenario=args.scenario.lower()
    protocol=ROOT/"docs"/"BENCHMARK_PROTOCOL.md"
    if not protocol.exists(): raise SystemExit("Register docs/BENCHMARK_PROTOCOL.md before running the benchmark")
    cfg=yaml.safe_load((ROOT/"config"/"benchmark.yaml").read_text(encoding="utf-8"))
    scales=cfg.get("scales")
    if not scales: raise SystemExit("Frozen global scales are missing; run scripts/05_calibrate_scales.py")
    qpath=ROOT/"derived_data"/"benchmark"/f"queries_{scenario}.parquet"
    if not qpath.exists(): raise SystemExit(f"Generate queries first: python scripts/04_generate_queries.py {scenario}")
    queries=pd.read_parquet(qpath)
    if args.query_limit: queries=queries.head(args.query_limit)
    radius=cfg["scenarios"]["core_radius_km"] if scenario=="core" else (cfg["scenarios"]["mid_radius_km"] if scenario=="mid" else None)
    preparation_started=time.perf_counter()
    net=PreparedNetwork.from_directory(ROOT/"derived_data"/"processed",scenario=scenario,radius_km=radius,
                                       boarding_buffer_s=30)
    preparation_s=time.perf_counter()-preparation_started
    profiles_frame=pd.read_csv(ROOT/"config"/"profiles.csv")
    profiles=[{k:float(row[col]) for k,col in zip(OBJECTIVES,profiles_frame.columns[1:])}
              for _,row in profiles_frame.iterrows()]
    nrep=int(args.repetitions or cfg["repetitions"]); cap=int(cfg["label_cap"])
    result_rows=[]; run_rows=[]; journey_file=ROOT/"results"/f"journeys_{scenario}.jsonl"
    (ROOT/"results").mkdir(exist_ok=True)
    if journey_file.exists(): journey_file.unlink()
    warmed=set()
    for qrow in queries.itertuples(index=False):
      qbase=(float(qrow.origin_lat),float(qrow.origin_lon)),(float(qrow.destination_lat),float(qrow.destination_lon)),int(qrow.departure_time)
      query_id=str(qrow.query_id)
      for objectives in OBJECTIVE_SETS:
        per_method={}; stats_per={}; exact_all={}
        q=Query(qbase[0],qbase[1],qbase[2],tuple(objectives),timeout_s=float(cfg["timeout_s"]),
                max_transfers=int(cfg["max_transfers"]))
        for algorithm in args.algorithms:
          if algorithm=="A1" and scenario!="core": continue
          reps=[]; memory=[]; calls=[]
          if algorithm not in warmed:
            warm_q=Query(q.origin,q.destination,q.departure_time,("T","C"),timeout_s=min(5.0,q.timeout_s))
            if algorithm=="A1": LabelSettingRouter(net,label_cap=min(cap,10000)).route(warm_q)
            elif algorithm=="A2": McRaptorRouter(net,label_cap=min(cap,10000)).route(warm_q)
            elif algorithm=="W": WeightSweepRouter(net,scales=scales,sweep_size=2,seed=int(cfg["seed"]),profiles=[],label_cap=min(cap,10000)).route(warm_q)
            else: WeightedAStarRouter(net,scales=scales,label_cap=min(cap,10000)).route(Query(q.origin,q.destination,q.departure_time,("T","C"),{"T":1,"C":0},timeout_s=min(5.0,q.timeout_s)))
            warmed.add(algorithm)
          for rep in range(nrep):
            tracemalloc.start(); start=time.perf_counter()
            if algorithm=="A1": result=LabelSettingRouter(net,label_cap=cap).route(q)
            elif algorithm=="A2": result=McRaptorRouter(net,label_cap=cap).route(q)
            elif algorithm=="W": result=WeightSweepRouter(net,scales=scales,sweep_size=int(cfg["sweep_size"]),
                    seed=int(cfg["seed"]),profiles=profiles,label_cap=cap).route(q)
            else:
              candidates=[]; bstats=[]
              for profile in profiles:
                qb=Query(q.origin,q.destination,q.departure_time,q.objectives,profile,q.disruptions,
                         q.start_state,q.max_transfers,q.timeout_s)
                br=WeightedAStarRouter(net,scales=scales,label_cap=cap).route(qb)
                candidates.extend(br.journeys); bstats.append(br.stats)
              from pumo.model import RouteResult
              result=RouteResult(candidates,{"runtime_s":sum(s.get("runtime_s",0) for s in bstats),
                "labels_generated":sum(s.get("labels_generated",0) for s in bstats),
                "labels_expanded":sum(s.get("labels_expanded",0) for s in bstats),
                "timed_out":any(s.get("timed_out") for s in bstats),
                "label_cap_hit":any(s.get("label_cap_hit") for s in bstats),
                "exact":all(s.get("exact",False) for s in bstats)})
            elapsed=time.perf_counter()-start; _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
            reps.append(result); memory.append(peak/1024**2); calls.append(elapsed)
            run_rows.append({"query_id":query_id,"scenario":scenario,"algorithm":algorithm,
              "objective_set":"".join(objectives),"objectives":",".join(objectives),"repetition":rep,
              "seed":int(qrow.seed),"distance_band":qrow.distance_band,
              "runtime_s":elapsed,"memory_peak_mib":peak/1024**2,
              "labels_generated":result.stats.get("labels_generated",0),"labels_expanded":result.stats.get("labels_expanded",0),
              "journeys_returned":len(result.journeys),"timed_out":result.stats.get("timed_out",False),
              "label_cap_hit":result.stats.get("label_cap_hit",False),"exact":result.stats.get("exact",False),
              "failure":bool(result.stats.get("timed_out",False) or result.stats.get("label_cap_hit",False) or not result.journeys),
              "total_runtime_s":result.stats.get("total_runtime_s",result.stats.get("runtime_s",elapsed)),
              "weight_vectors_run":result.stats.get("weight_vectors_run",0)})
          per_method[algorithm]=reps[-1].journeys if reps else []
          stats_per[algorithm]=reps[-1].stats if reps else {}
          exact_all[algorithm]=all(r.stats.get("exact",False) for r in reps)
          for j,journey in enumerate(per_method[algorithm]):
            with journey_file.open("a",encoding="utf-8") as f:
              f.write(json.dumps({"query_id":query_id,"objective_set":"".join(objectives),
                "algorithm":algorithm,"journey_index":j,"vector":journey.vector,
                "legs":[{"mode":leg.mode,"from":leg.from_ref,"to":leg.to_ref,"dep":leg.dep_time,
                         "arr":leg.arr_time,"segments":leg.segment_ids} for leg in journey.legs]})+"\n")
          result_rows.append({"query_id":query_id,"scenario":scenario,"algorithm":algorithm,
            "objective_set":"".join(objectives),"objectives":",".join(objectives),"repetitions":nrep,
            "runtime_median_s":float(pd.Series(calls).median()),"memory_peak_mib":max(memory,default=0),
            "labels_generated":stats_per[algorithm].get("labels_generated",0),
            "labels_expanded":stats_per[algorithm].get("labels_expanded",0),
            "journeys_returned":len(per_method[algorithm]),"timed_out":stats_per[algorithm].get("timed_out",False),
            "label_cap_hit":stats_per[algorithm].get("label_cap_hit",False),"exact":stats_per[algorithm].get("exact",False),
            "total_runtime_s":stats_per[algorithm].get("total_runtime_s",stats_per[algorithm].get("runtime_s",0)),
            "weight_vectors_run":stats_per[algorithm].get("weight_vectors_run",0)})
        a1_exact = ("A1" in stats_per and exact_all.get("A1",False))
        if a1_exact:
            reference=per_method["A1"]; reference_type="EXACT_A1"
        else:
            from pumo.pareto import pareto_filter
            candidates=[j for a in ("A2","W","B0") for j in per_method.get(a,[])]
            reference=pareto_filter(candidates,lambda j:tuple(j.vector.get(k,0) for k in objectives))
            reference_type="BEST_KNOWN"
        for row in result_rows[-len(stats_per):]:
          if row["query_id"]!=query_id or row["objective_set"]!="".join(objectives): continue
          method=row["algorithm"]; returned=per_method.get(method,[])
          quality=result_metrics(returned,reference,objectives,global_scales=scales,profiles=profiles,
              seed=int(cfg["seed"]),mc_samples=int(cfg["monte_carlo_samples"]))
          row.update(quality); row["reference_type"]=reference_type
    quality_lookup={(r["query_id"],r["objective_set"],r["algorithm"]):r for r in result_rows}
    for row in run_rows:
        quality=quality_lookup.get((row["query_id"],row["objective_set"],row["algorithm"]),{})
        for key in ("recall","dominated_fraction","hypervolume_ratio","linear_profile_regret","chebyshev_regret","reference_type"):
            if key in quality: row[key]=quality[key]
    raw=pd.DataFrame(run_rows)
    raw.to_csv(ROOT/"results"/"raw_runs.csv",index=False)
    query_summary=raw.groupby(["scenario","query_id","algorithm","objective_set"],as_index=False).agg(
        query_runtime_s=("runtime_s","median"),memory_peak_mib=("memory_peak_mib","max"),
        journeys_returned=("journeys_returned","median"),labels_generated=("labels_generated","median"),
        failure=("failure","max"),timed_out=("timed_out","max"),label_cap_hit=("label_cap_hit","max"))
    summary=query_summary.groupby(["scenario","algorithm","objective_set"],as_index=False).agg(
        median_runtime_s=("query_runtime_s","median"),p95_runtime_s=("query_runtime_s",lambda s:s.quantile(.95)),
        median_memory_mib=("memory_peak_mib","median"),median_journeys=("journeys_returned","median"),
        mean_labels_generated=("labels_generated","mean"),failure_rate=("failure","mean"))
    for metric in ("recall","dominated_fraction","hypervolume_ratio","linear_profile_regret","chebyshev_regret"):
        if metric in raw: summary=summary.merge(raw.groupby(["scenario","algorithm","objective_set"])[metric].mean().rename(metric).reset_index(),
                                                 on=["scenario","algorithm","objective_set"],how="left")
    summary.to_csv(ROOT/"results"/f"summary_{scenario}.csv",index=False)
    try: revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception: revision=None
    try:
        import psutil
        memory_total=psutil.virtual_memory().total
    except Exception: memory_total=None
    package_versions={}
    for package in ("numpy","pandas","pyarrow","geopandas","shapely","pyproj","scipy","pyrosm","psutil","matplotlib"):
        try: package_versions[package]=version(package)
        except PackageNotFoundError: package_versions[package]=None
    metadata={"scenario":scenario,"query_count":len(queries),"objective_sets":["".join(o) for o in OBJECTIVE_SETS],
        "repetitions":nrep,"seed":cfg["seed"],"python":platform.python_version(),"platform":platform.platform(),
        "cpu":platform.processor(),"cpu_count":os.cpu_count(),"ram_bytes":memory_total,"git_revision":revision,
        "package_versions":package_versions,"network_preprocessing_s":preparation_s,"scales":scales,
        "configuration":cfg}
    (ROOT/"results"/f"run_metadata_{scenario}.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(summary.to_string(index=False)); print(json.dumps(metadata,indent=2))

if __name__=="__main__": main()
