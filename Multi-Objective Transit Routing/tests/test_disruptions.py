from pumo.bench.disruptions import choose_disruption
from pumo.model import Journey,Leg


def test_synthetic_disruption_is_seeded_and_on_followed_route():
    journey=Journey([
        Leg("WALK","0","1",0,600,["5"],{"T":600,"W":100}),
        Leg("BUS","1","2",700,1200,["trip-A:1>2"],{"T":600,"C":1}),
    ],{"T":1200,"W":100,"C":1})
    a=choose_disruption(journey,650,seed=42)
    b=choose_disruption(journey,650,seed=42)
    assert a[0]==b[0]
    assert a[1].current_position=="node:1"
    assert a[0].trip_ids==("trip-A",)
    assert a[0].time_penalty_s in (300,600,1200)
