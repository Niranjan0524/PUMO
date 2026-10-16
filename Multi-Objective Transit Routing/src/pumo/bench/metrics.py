"""Quality metrics for multi-objective result sets."""
from __future__ import annotations
import numpy as np

from pumo.pareto import dominates, epsilon_match, pareto_indices
from pumo.objectives import weighted_score, chebyshev_score


def _matrix(items, objectives):
    if not items: return np.empty((0,len(objectives)),dtype=float)
    return np.asarray([[float(item.vector.get(k,0.0)) if hasattr(item,"vector") else float(item.get(k,0.0))
                        for k in objectives] for item in items],dtype=float)


def normalize_pair(returned, reference, objectives, global_scales):
    r=_matrix(returned,objectives); ref=_matrix(reference,objectives)
    if not len(r) or not len(ref): return r,ref
    if len(ref)==1:
        scales=np.asarray([float(global_scales[k]) for k in objectives])
        lo=np.zeros(len(objectives)); denom=scales
    else:
        lo=ref.min(axis=0); denom=ref.max(axis=0)-lo
        fallback=np.asarray([float(global_scales[k]) for k in objectives])
        denom=np.where(denom>0,denom,fallback)
    return (r-lo)/denom,(ref-lo)/denom


def hypervolume(points, reference_point=None):
    """Exact dominated hypervolume for 1–4 minimized dimensions."""
    points=np.asarray(points,dtype=float)
    if points.ndim!=2: raise ValueError("points must be a 2-D matrix")
    d=points.shape[1]
    if d==0: return 0.0
    if d>4: raise ValueError("exact hypervolume supports at most 4 objectives")
    ref=np.asarray(reference_point if reference_point is not None else np.full(d,1.1),dtype=float)
    if ref.shape!=(d,): raise ValueError("reference point dimension mismatch")
    points=points[np.all(points<ref,axis=1)]
    if not len(points): return 0.0
    points=points[pareto_indices(points.tolist())]
    return _hv_recursive(points,ref)


def _hv_recursive(points,ref):
    d=points.shape[1]
    if d==1: return max(0.0,float(ref[0]-np.min(points[:,0])))
    xs=sorted(set(points[:,0].tolist()))
    total=0.0; active=[]; previous=xs[0]
    for x in xs:
        if x>previous and active:
            total+=(x-previous)*_hv_recursive(np.asarray(active)[:,1:],ref[1:])
        active.extend(points[points[:,0]==x].tolist())
        previous=x
    if active and ref[0]>previous:
        total+=(ref[0]-previous)*_hv_recursive(np.asarray(active)[:,1:],ref[1:])
    return float(total)


def hypervolume_mc(points, reference_point=None, *, samples=100_000, seed=20261016):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2: raise ValueError("points must be a 2-D matrix")
    d=points.shape[1]
    ref=np.asarray(reference_point if reference_point is not None else np.full(d,1.1),dtype=float)
    rng=np.random.default_rng(seed); sample=rng.random((int(samples),d))*ref
    dominated=np.zeros(len(sample),dtype=bool)
    for point in points:
        dominated |= np.all(point<=sample,axis=1)
    return float(np.prod(ref)*dominated.mean())


def result_metrics(returned, reference, objectives, *, global_scales, profiles=None,
                   epsilon=.01, seed=20261016, mc_samples=100_000):
    r,ref=normalize_pair(returned,reference,objectives,global_scales)
    metric={"n_returned":len(r),"n_reference":len(ref)}
    if len(r) and len(ref):
        metric["recall"]=sum(any(epsilon_match(rv,refv,relative=epsilon) for rv in ref) for rv in ref)/len(ref)
        metric["dominated_fraction"]=sum(any(dominates(refv,rv) for refv in ref) for rv in r)/len(r)
        reference_point=np.full(len(objectives),1.1)
        if len(objectives)<=4:
            hv_ref=hypervolume(ref,reference_point); hv_out=hypervolume(r,reference_point)
            metric["hypervolume_method"]="exact" if len(objectives)<=4 else "monte_carlo"
        else:
            hv_ref=hypervolume_mc(ref,reference_point,samples=mc_samples,seed=seed)
            hv_out=hypervolume_mc(r,reference_point,samples=mc_samples,seed=seed)
            metric["hypervolume_method"]="seeded_monte_carlo"
        metric["hypervolume_ratio"]=hv_out/hv_ref if hv_ref>0 else float("nan")
    else:
        metric.update(recall=float("nan"),dominated_fraction=float("nan"),hypervolume_ratio=float("nan"))
    if profiles:
        linear=[]; cheb=[]
        raw_c=_matrix(returned,objectives); raw_ref=_matrix(reference,objectives)
        scales=np.asarray([float(global_scales[k]) for k in objectives])
        raw_c=raw_c/scales; raw_ref=raw_ref/scales
        for weights in profiles:
            if not len(raw_c) or not len(raw_ref): continue
            norm_c=[{k:float(v) for k,v in zip(objectives,row)} for row in raw_c]
            norm_ref=[{k:float(v) for k,v in zip(objectives,row)} for row in raw_ref]
            ls=lambda x:weighted_score(x,weights,{k:1.0 for k in objectives},objectives)
            cs=lambda x:chebyshev_score(x,weights,{k:1.0 for k in objectives},objectives)
            best_l=min(map(ls,norm_c)); ref_l=min(map(ls,norm_ref))
            best_c=min(map(cs,norm_c)); ref_c=min(map(cs,norm_ref))
            linear.append((best_l-ref_l)/max(abs(ref_l),1e-12))
            cheb.append((best_c-ref_c)/max(abs(ref_c),1e-12))
        metric["linear_profile_regret"]=float(np.mean(linear)) if linear else float("nan")
        metric["chebyshev_regret"]=float(np.mean(cheb)) if cheb else float("nan")
    return metric
