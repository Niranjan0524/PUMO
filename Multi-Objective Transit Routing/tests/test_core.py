from pumo.objectives import OBJECTIVES, objective_vector, sum_leg_vectors, weighted_score
from pumo.pareto import dominates, epsilon_match, pareto_filter, pareto_indices
from pumo.model import Journey, Leg, RouteResult, Query, validate_journey, validate_result
from pumo.objectives import normalize
from pumo.timetable import gtfs_time_seconds


def test_dominance_ties_duplicates_and_strictness():
    assert dominates((1, 2), (2, 2))
    assert not dominates((1, 2), (1, 2))
    assert not dominates((2, 2), (1, 2))
    assert pareto_indices([(1, 2), (1, 2), (2, 1), (3, 3)]) == [0, 2]


def test_epsilon_match_relative():
    assert epsilon_match((100, 10), (100.5, 10.05), relative=.01)
    assert not epsilon_match((100,), (103,), relative=.01)


def test_weighted_score_and_objective_order():
    v = {"T": 10, "C": 4, "X": 1}
    assert objective_vector(v, ("T", "C", "X")) == (10, 4, 1)
    assert weighted_score(v, {"T": .5, "C": .5}, {"T": 10, "C": 4}, ("T", "C")) == 1
    assert OBJECTIVES == ("T", "C", "X", "W", "E", "S")


def test_journey_validation_recomputes_vector():
    legs = [Leg("WALK", "a", "b", 10, 20, ["e1"], {"T": 10, "W": 5}),
            Leg("BUS", "b", "c", 25, 40, ["t1"], {"T": 20, "C": 1, "X": 0})]
    journey = Journey(legs, {"T": 30, "C": 1, "X": 0, "W": 5})
    validate_journey(journey)
    assert sum_leg_vectors(legs)["T"] == 30
    journey.vector["W"] = 6
    try:
        validate_journey(journey)
    except ValueError as exc:
        assert "disagrees" in str(exc)
    else:
        raise AssertionError("invalid aggregate vector accepted")


def test_pareto_filter_preserves_stable_order():
    values = [{"v": (1, 3)}, {"v": (2, 2)}, {"v": (3, 1)}, {"v": (4, 4)}]
    assert pareto_filter(values, lambda x: x["v"]) == values[:3]


def test_unsupported_pareto_point_survives_shared_filter():
    points = [(1, 8), (4, 6), (8, 1), (5, 8)]
    survivors = [points[i] for i in pareto_indices(points)]
    assert survivors == points[:3]
    assert (4, 6) not in [(1, 8), (8, 1)]


def test_objective_normalization_rejects_bad_scales_and_nonfinite_values():
    assert normalize({"T": 20, "C": 4}, {"T": 10, "C": 2}, ("T", "C")) == {"T": 2, "C": 2}
    for scales in ({"T": 0}, {"T": float("inf")}, {}):
        try:
            normalize({"T": 1}, scales, ("T",))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid scale accepted")


def test_gtfs_time_keeps_hours_beyond_midnight():
    assert gtfs_time_seconds("25:10:05") == 90_605
    for bad in ("12:60:00", "-1:00:00", "12:00", "x:y:z"):
        try:
            gtfs_time_seconds(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid time accepted: {bad}")


def test_result_validator_rejects_dominated_and_infeasible_journeys():
    good = Journey([Leg("WALK", "a", "b", 0, 10, ["w"], {"T": 10, "C": 0})], {"T": 10, "C": 0})
    dominated = Journey([Leg("WALK", "a", "b", 0, 12, ["w2"], {"T": 12, "C": 1})], {"T": 12, "C": 1})
    try:
        validate_result(RouteResult([good, dominated]), objectives=("T", "C"))
    except ValueError as exc:
        assert "dominated" in str(exc)
    else:
        raise AssertionError("dominated result accepted")
    disconnected = Journey([Leg("WALK", "a", "b", 0, 10, ["w"], {"T": 10}) ,
                            Leg("BUS", "c", "d", 9, 20, ["b"], {"T": 11})], {"T": 21})
    try:
        validate_journey(disconnected)
    except ValueError:
        pass
    else:
        raise AssertionError("disconnected/backwards journey accepted")


def test_query_validates_objective_and_search_limits():
    Query((0, 0), (1, 1), 0, ("T",))
    for kwargs in ({"objectives": ()}, {"objectives": ("Q",)}, {"max_transfers": -1}, {"timeout_s": 0}):
        try:
            Query((0, 0), (1, 1), 0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid query accepted: {kwargs}")
