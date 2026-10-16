"""Reproducible synthetic disruptions selected from a followed journey."""
from __future__ import annotations
import numpy as np

from pumo.model import Disruption, JourneyState


def choose_disruption(journey, activation_time: int, *, seed: int = 20261016,
                      duration_s: int = 3600, safety_penalty: float = 1.0):
    """Choose an event on the journey active at/after the mid-journey time.

    Returns ``(event, state, baseline_suffix)``. If activation falls in a gap,
    the next leg is selected. A state is represented at the leg's starting node.
    """
    rng=np.random.default_rng(seed)
    candidate_indices=[i for i,leg in enumerate(journey.legs) if leg.arr_time>=activation_time]
    if not candidate_indices: return None,None,[]
    i=candidate_indices[0]; leg=journey.legs[i]
    activation=max(int(activation_time),int(leg.dep_time))
    suffix=journey.legs[i:]
    if leg.mode=="WALK":
        ids=[]
        for ref in leg.segment_ids:
            try: ids.append(int(ref))
            except (ValueError,TypeError): pass
        if not ids: return None,None,suffix
        edge=int(rng.choice(ids)); event_type=str(rng.choice(["ROAD_CLOSURE","FLOOD","LIGHTING_OUTAGE"]))
        event=Disruption(f"synthetic-{seed}",event_type,activation,activation+duration_s,
            edge_ids=(edge,),closure=event_type in {"ROAD_CLOSURE","FLOOD"},
            safety_penalty=safety_penalty,seed=seed)
    elif leg.mode in {"BUS","METRO"}:
        trip=leg.segment_ids[0].split(":",1)[0]
        delay=int(rng.choice([300,600,1200]))
        event=Disruption(f"synthetic-{seed}",f"{leg.mode}_DELAY",activation,activation+duration_s,
            trip_ids=(trip,),time_penalty_s=delay,seed=seed)
    else:
        return None,None,suffix
    state=JourneyState(f"node:{leg.from_ref}",activation,leg.mode,tuple(journey.legs[:i]))
    return event,state,suffix


def shared_segment_ratio(before, after):
    left={segment for leg in before for segment in leg.segment_ids}
    right={segment for leg in after for segment in leg.segment_ids}
    if not left and not right: return 1.0
    return len(left & right)/max(1,len(left | right))
