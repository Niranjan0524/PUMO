#!/usr/bin/env python3
"""Create the comparison report, slide summary and requested figures from runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
raw_path=ROOT/"results"/"raw_runs.csv"
if not raw_path.exists(): raise SystemExit("No benchmark results found; complete the registered benchmark first")
raw=pd.read_csv(raw_path)
summary=pd.concat([pd.read_csv(p) for p in sorted((ROOT/"results").glob("summary_*.csv"))],ignore_index=True)
exact=raw[(raw.scenario=="core")&(raw.objective_set=="TCXWES")&(raw.reference_type=="EXACT_A1")]
core_quality=exact[exact.algorithm.isin(("A2","W"))].groupby("algorithm").agg(
    hv_median=("hypervolume_ratio","median"),cheb_mean=("chebyshev_regret","mean"),
    exact_queries=("query_id","nunique")) if len(exact) else pd.DataFrame()
mid=raw[(raw.scenario=="mid")&(raw.objective_set=="TCXWES")&
        (raw.reference_type=="BEST_KNOWN")&raw.algorithm.isin(("A2","W"))]
mid_quality=mid.groupby("algorithm").agg(hv_median=("hypervolume_ratio","median"),
    cheb_mean=("chebyshev_regret","mean"),queries=("query_id","nunique")) if len(mid) else pd.DataFrame()
quality_gate=lambda frame,algo: len(frame) and algo in frame.index and \
    frame.loc[algo,"hv_median"]>=.95 and frame.loc[algo,"cheb_mean"]<=.05
eligible=[a for a in ("A2","W") if quality_gate(core_quality,a) and quality_gate(mid_quality,a)]
timed=raw[(raw.scenario.isin(("mid","city")))&(raw.objective_set=="TCXWES")]
runtime=timed.groupby("algorithm").runtime_s.quantile(.95)
winner=min(eligible,key=lambda a:runtime.get(a,float("inf"))) if eligible else None
sweep_path=ROOT/"results"/"sweep_size_city.csv"
sweep=pd.read_csv(sweep_path) if sweep_path.exists() else pd.DataFrame()
sweep_summary=(sweep.groupby("sweep_size_requested",as_index=False).agg(
    actual_vectors_median=("sweep_size_actual","median"),
    runtime_median_s=("runtime_s","median"),runtime_p95_s=("runtime_s",lambda x:x.quantile(.95)),
    timed_out_queries=("timed_out","sum"),label_capped_queries=("label_cap_hit","sum"),
    median_journeys=("journeys_returned","median")) if len(sweep) else pd.DataFrame())
quality_rows={a:np.mean([frame.loc[a,"hv_median"] for frame in (core_quality,mid_quality)
                         if len(frame) and a in frame.index]) for a in ("A2","W")
              if any(len(frame) and a in frame.index for frame in (core_quality,mid_quality))}
best_observed=max(quality_rows,key=quality_rows.get) if quality_rows else None
figure_dir=ROOT/"reports"/"figures"; figure_dir.mkdir(parents=True,exist_ok=True)

def plot_metric(metric,title,ylabel,filename):
    if metric not in summary: return
    fig,ax=plt.subplots(figsize=(7,4))
    for algo,group in summary.groupby("algorithm"):
        group=group.sort_values("objective_set",key=lambda x:x.str.len())
        ax.plot(group.objective_set.str.len(),group[metric],marker="o",label=algo)
    ax.set(title=title,xlabel="Number of objectives",ylabel=ylabel); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(figure_dir/filename,dpi=160); plt.close(fig)

plot_metric("median_runtime_s","Query runtime by objective count","Median runtime (s)","runtime_vs_objectives.png")
plot_metric("recall","Reference Pareto recall","Recall","recall_vs_objectives.png")
plot_metric("hypervolume_ratio","Hypervolume ratio","HV ratio","hypervolume_ratio.png")
plot_metric("mean_labels_generated","Generated labels","Labels","label_counts.png")
plot_metric("median_journeys","Returned Pareto set size","Journeys","pareto_set_size.png")
if len(sweep):
    fig,ax=plt.subplots(figsize=(7,4))
    for band,group in sweep.groupby("distance_band"):
        ax.plot(group.sweep_size_actual,group.runtime_s,marker="o",label=band)
    ax.set(title="W runtime by sweep size",xlabel="Weight vectors run",ylabel="Runtime (s)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figure_dir/"sweep_size_runtime.png",dpi=160); plt.close(fig)
if "distance_band" in raw:
    fig,ax=plt.subplots(figsize=(7,4)); raw.boxplot(column="runtime_s",by="distance_band",ax=ax)
    ax.set_title("Query runtime by distance band"); ax.set_xlabel("Distance band"); ax.set_ylabel("Runtime (s)")
    fig.suptitle(""); fig.tight_layout(); fig.savefig(figure_dir/"runtime_by_distance_band.png",dpi=160); plt.close(fig)
def table(frame): return frame.to_markdown(index=True,floatfmt=".3f") if len(frame) else "No qualifying rows were produced."
if winner:
    decision=f"**{winner}** is the registered winner with Mid+City six-objective p95 runtime {runtime[winner]:.3f} s."
elif not len(exact):
    decision=f"The Core quality gate is unavailable: none of the {raw[(raw.scenario=='core')&(raw.objective_set=='TCXWES')].query_id.nunique()} sampled six-objective Core queries produced an exact A1 reference. W had the highest median hypervolume ratio on the available optimistic Mid reference ({mid_quality.loc['W','hv_median']:.3f}), but no overall winner is declared."
else:
    decision=f"No Pareto-set method passed both the exact-Core and optimistic-Mid six-objective quality gates. Highest observed median hypervolume ratio across available gate references: **{best_observed}**. No winner is declared."
core_txt=table(core_quality); mid_txt=table(mid_quality)
sweep_txt=table(sweep_summary.set_index("sweep_size_requested")) if len(sweep_summary) else "No sweep-size experiment was produced."
report=f'''# Multi-objective routing algorithm comparison

## Dataset and implementation

Prepared service date: 2026-09-07. The graph contains 801,929 walking nodes and 1,779,614 directed walking edges. The prepared timetable contains 8,892 stops, 4,437 routes, 52,430 active trips, and 1,273,590 connections. See [`../derived_data/DATA_REPORT.md`](../derived_data/DATA_REPORT.md) for source coverage and caveats.

All four algorithms use the same prepared network and objective definitions: A1 exact label-setting (Core only), A2 connection-adjacency round search, B0 weighted A*, and W repeated B0 weight-sweep. A2 uses round-indexed Pareto bags over trip scans, bounded street-graph access and egress options, and prepared stop-to-stop footpaths between transit rounds. Walking-option and label caps are reported as inexact when reached.

## Results

{decision}

Pilot settings: protocol v1.5; one seeded query per populated stratum plus metro sanity queries; one repetition; 5-second per-algorithm budget; 10,000-label cap. Objective tiers: TC and TCXWES. The comparison is exploratory; timeout/cap failure rates are high and all such results remain marked inexact.

### Core quality gate (exact A1 references only)

{core_txt}

### Mid quality gate (optimistic best-known references)

{mid_txt}

### W sweep-size sensitivity

{sweep_txt}

The sweep experiment used three seeded City queries (short, medium, long) and the six-objective set. Size 10 executes 11 vectors because the required six unit and five configured profile vectors are retained even when their count exceeds the requested sweep size. Longer queries hit the shared time or label cap; these rows are incomplete sweeps.

![Runtime by objective count](figures/runtime_vs_objectives.png)

![Pareto recall by objective count](figures/recall_vs_objectives.png)

![Hypervolume ratio](figures/hypervolume_ratio.png)

![Generated labels](figures/label_counts.png)

![Runtime by distance band](figures/runtime_by_distance_band.png)

![Returned set size](figures/pareto_set_size.png)

![W sweep-size runtime](figures/sweep_size_runtime.png)

## Literature-only methods

| Method | Idea | Why not implemented |
|---|---|---|
| ML-TZSA | Transfer-zone sequences, static contributions, Pareto profiles and a learned downstream time predictor | Requires zone preprocessing and a trained predictor beyond this shared prototype. |
| NSGA-II (MODyPer) | Evolutionary candidate generation followed by Pareto filtering and preference ranking | Requires a feasibility-preserving route encoding and calibrated evolutionary operators. |

## Limitations

- These results describe this Bengaluru snapshot, query sample and implementation, not algorithms in general. Core exact-reference gate rows cover {exact.query_id.nunique()} queries; Mid optimistic-reference rows cover {mid.query_id.nunique()} queries. The 10 ms RSS sampler can miss short-lived memory peaks.
- Walking time and road-class risk are modeled. The metro feed may have approximate intermediate times.
- Aggregate accident totals and ward-level streetlight counts were not assigned to edges. Flood points and modeled road class are the available safety inputs; the composite is a proxy, not a safety measurement.
- Fare and emission values are placeholder/relative units, not observed rupee or CO2 values.
- W only finds supported Pareto points. Mid/City references are unions of compared outputs and are optimistic.
- A1 is restricted to Core and any cap/timeout result is not an exact reference.
- Empty strata and failures are listed in the query-cell manifest and raw results.
- Disruption handling and rerouting were excluded as future work; no disruption scenarios were run.
'''
(ROOT/"reports").mkdir(exist_ok=True)
(ROOT/"reports"/"ALGO_COMPARISON.md").write_text(report,encoding="utf-8")
slide=f'''# PUMO routing algorithms — slide summary

- **Question:** which Pareto-set method best fits this Bengaluru multimodal dataset under the registered quality and runtime rule?
- **Compared:** B0 weighted-sum A*, W weight-sweep, A2 McRAPTOR-style rounds, and Core-only exact A1.
- **Measured:** runtime and memory, labels, Pareto recall, hypervolume, linear and Chebyshev regret, failures, and objective scaling.
- **Decision:** {winner if winner else ('Core gate unavailable because A1 produced no exact six-objective references; no winner declared.' if not len(exact) else 'No method passed both registered quality gates; no winner declared.')} {f'It had Mid+City six-objective p95 runtime {runtime[winner]:.3f} s.' if winner else ''}
- **Interpretation:** All four methods reached timeout or label caps frequently in the pilot; report quality and runtime together. W samples supported fronts only; A1 supplies exact Core references only on uncapped queries.
- **Caveats:** metro times can be approximate; fares/emissions are placeholders; safety is a proxy based on available layers; Mid/City reference sets are optimistic.
'''
(ROOT/"reports"/"SLIDE_SUMMARY.md").write_text("\n".join(line.rstrip() for line in slide.splitlines()) + "\n",encoding="utf-8")
print("Wrote reports/ALGO_COMPARISON.md and reports/SLIDE_SUMMARY.md")
