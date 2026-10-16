"""Seeded stratified origin-destination query generation."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from scipy.spatial import cKDTree


def generate_queries(prepared_dir, scenario, *, seed=20261016, per_cell=None,
                     access_radius_m=600):
    p=Path(prepared_dir); stops=pd.read_parquet(p/"stops.parquet")
    nodes=gpd.read_parquet(p/"walking_nodes.parquet")
    conns=pd.read_parquet(p/"connections.parquet")
    tf=Transformer.from_crs("EPSG:4326","EPSG:32643",always_xy=True)
    stop_xy=np.column_stack(tf.transform(stops.lon.to_numpy(),stops.lat.to_numpy()))
    stop_tree=cKDTree(stop_xy); rng=np.random.default_rng(seed)
    radius={"core":3000,"mid":8000,"city":float("inf")}[scenario.lower()]
    neighborhoods=stop_tree.query_ball_point(stop_xy,r=3000)
    center=stop_xy[int(np.argmax([len(x) for x in neighborhoods]))]
    node_xy=np.column_stack(tf.transform(nodes.geometry.x.to_numpy(),nodes.geometry.y.to_numpy()))
    keep=np.ones(len(nodes),dtype=bool) if not np.isfinite(radius) else np.linalg.norm(node_xy-center,axis=1)<=radius
    local_xy=node_xy[keep]; local_nodes=nodes.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    local_tree=cKDTree(local_xy)
    nearby_counts=np.asarray(stop_tree.query_ball_point(local_xy,r=access_radius_m,return_length=True))
    richness_order=np.argsort(nearby_counts,kind="stable")
    richness_size=max(1,len(richness_order)//3)
    richness_nodes={
        "low":set(richness_order[:richness_size].tolist()),
        "high":set(richness_order[-richness_size:].tolist()),
    }
    departure=conns.dep_time.dropna().to_numpy(dtype=np.int64)
    bands={"morning_peak":(5*3600,10*3600),"midday":(10*3600,16*3600),
           "evening_peak":(16*3600,20*3600),"late_evening":(20*3600,30*3600)}
    available={name:departure[(departure>=lo)&(departure<hi)] for name,(lo,hi) in bands.items()}
    available={name:values for name,values in available.items() if len(values)}
    dist_bands={"short":(1000,3000),"medium":(3000,8000),"long":(8000,float("inf"))}
    defaults={"core":10,"mid":6,"city":4}; need=int(per_cell if per_cell is not None else defaults[scenario.lower()])
    queries=[]; cell_counts={}
    for distance_name,(dmin,dmax) in dist_bands.items():
      if np.isfinite(radius) and dmin >= 2*radius: continue
      search_radius=min(dmax,2*radius) if np.isfinite(dmax) and np.isfinite(radius) else (2*radius if np.isfinite(radius) else dmax)
      for transit_class in ("low","high"):
       for time_band,times in available.items():
        key=f"{distance_name}|{transit_class}|{time_band}"; made=0; attempts=0
        while made<need and attempts<max(200,need*100):
            attempts+=1; oi=int(rng.integers(len(local_xy))); origin=local_xy[oi]
            if oi not in richness_nodes[transit_class]: continue
            candidates=local_tree.query_ball_point(origin,search_radius)
            if len(candidates)<2: continue
            di=int(rng.choice(candidates));
            if di==oi: continue
            dist=float(np.linalg.norm(local_xy[di]-origin))
            if not dmin<=dist<dmax: continue
            if di not in richness_nodes[transit_class]: continue
            a=local_nodes.iloc[oi]; b=local_nodes.iloc[di]
            nearest_a=int(stops.iloc[int(stop_tree.query(origin)[1])].stop_idx)
            nearest_b=int(stops.iloc[int(stop_tree.query(local_xy[di])[1])].stop_idx)
            queries.append({"query_id":f"{scenario.lower()}-{len(queries):05d}","scenario":scenario,
                "origin_lat":float(a.geometry.y),"origin_lon":float(a.geometry.x),"origin_node_idx":int(a.node_idx),"nearest_origin_stop_idx":nearest_a,
                "destination_lat":float(b.geometry.y),"destination_lon":float(b.geometry.x),"destination_node_idx":int(b.node_idx),"nearest_destination_stop_idx":nearest_b,
                "departure_time":int(rng.choice(times)),"distance_m":dist,"distance_band":distance_name,
                "transit_richness":transit_class,"departure_band":time_band,"seed":seed})
            made+=1
        cell_counts[key]={"requested":need,"generated":made,"attempts":attempts}
    # Hand-picked direct metro-segment checks within the chosen scenario region.
    metro_st=conns[conns.agency.eq("METRO")]
    metro_region=np.ones(len(stops),dtype=bool) if not np.isfinite(radius) else np.linalg.norm(stop_xy-center,axis=1)<=radius
    allowed=set(stops.loc[metro_region,"stop_idx"].astype(int))
    metro_st=metro_st[metro_st.from_stop.isin(allowed)&metro_st.to_stop.isin(allowed)]
    for row in metro_st.drop_duplicates("trip_id").head(5).itertuples(index=False):
        a=stops[stops.stop_idx.eq(int(row.from_stop))].iloc[0]
        b=stops[stops.stop_idx.eq(int(row.to_stop))].iloc[0]
        queries.append({"query_id":f"{scenario.lower()}-metro-{len(queries):05d}","scenario":scenario,
            "origin_lat":float(a.lat),"origin_lon":float(a.lon),"origin_node_idx":int(a.node_idx),"nearest_origin_stop_idx":int(a.stop_idx),
            "destination_lat":float(b.lat),"destination_lon":float(b.lon),"destination_node_idx":int(b.node_idx),"nearest_destination_stop_idx":int(b.stop_idx),
            "departure_time":int(row.dep_time),"distance_m":float(np.linalg.norm(stop_xy[int(a.name)]-stop_xy[int(b.name)])),
            "distance_band":"metro_sanity","transit_richness":"high","departure_band":"published_timetable","seed":seed})
    result=pd.DataFrame(queries)
    out=p.parent/"benchmark"; out.mkdir(parents=True,exist_ok=True)
    result.to_parquet(out/f"queries_{scenario.lower()}.parquet",index=False)
    import json
    (out/f"query_cells_{scenario.lower()}.json").write_text(json.dumps(cell_counts,indent=2),encoding="utf-8")
    return result,cell_counts
