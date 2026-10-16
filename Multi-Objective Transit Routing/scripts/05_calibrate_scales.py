#!/usr/bin/env python3
"""Calibrate and freeze B0/W objective scales on Core T-fastest routes."""
import argparse, json, sys, yaml
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pumo.algos.weighted_astar import WeightedAStarRouter
from pumo.model import Query
from pumo.network import PreparedNetwork
from pumo.objectives import OBJECTIVES

ap=argparse.ArgumentParser(); ap.add_argument("--query-limit",type=int,default=10); ap.add_argument("--timeout",type=float,default=5.0); args=ap.parse_args()
p=ROOT/"derived_data"/"processed"; qpath=ROOT/"derived_data"/"benchmark"/"queries_core.parquet"
if not qpath.exists(): raise SystemExit("Generate Core queries first: python scripts/04_generate_queries.py core")
cfg=yaml.safe_load((ROOT/"config"/"benchmark.yaml").read_text(encoding="utf-8"))
queries=pd.read_parquet(qpath)
if args.query_limit: queries=queries.head(args.query_limit)
net=PreparedNetwork.from_directory(p,scenario="core",radius_km=cfg["scenarios"]["core_radius_km"],
                                   boarding_buffer_s=30)
router=WeightedAStarRouter(net,scales={k:1.0 for k in OBJECTIVES})
observed={k:[] for k in OBJECTIVES}; failures=0
for row in queries.itertuples(index=False):
    q=Query((float(row.origin_lat),float(row.origin_lon)),(float(row.destination_lat),float(row.destination_lon)),
        int(row.departure_time),("T",),{"T":1.0},timeout_s=min(args.timeout,float(cfg["timeout_s"])))
    result=router.route(q)
    if not result.journeys: failures+=1; continue
    for name in OBJECTIVES: observed[name].append(float(result.journeys[0].vector.get(name,0.0)))
scales={k:max(1.0,float(np.percentile(v,95))) if v else 1.0 for k,v in observed.items()}
cfg["scales"]=scales
(ROOT/"config"/"benchmark.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
metadata={"method":"95th percentile of each objective over T-only B0 fastest journeys",
    "query_count":len(queries),"successful_queries":len(queries)-failures,"failed_queries":failures,
    "scales":scales,"query_seed":int(queries.seed.iloc[0]) if len(queries) else None}
(ROOT/"derived_data"/"scale_calibration.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
print(json.dumps(metadata,indent=2))
