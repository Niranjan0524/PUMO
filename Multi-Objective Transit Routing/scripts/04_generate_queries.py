#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pumo.bench.scenarios import generate_queries

parser=argparse.ArgumentParser()
parser.add_argument("scenario",choices=["core","mid","city"])
parser.add_argument("--seed",type=int,default=20261016)
parser.add_argument("--per-cell",type=int)
args=parser.parse_args()
queries,cells=generate_queries(ROOT/"derived_data"/"processed",args.scenario,seed=args.seed,per_cell=args.per_cell)
print(json.dumps({"queries":len(queries),"cells":cells},indent=2))
