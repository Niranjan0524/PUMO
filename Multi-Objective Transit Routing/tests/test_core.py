from pumo.objectives import OBJECTIVES, objective_vector, sum_leg_vectors, weighted_score
from pumo.pareto import dominates, epsilon_match, pareto_filter, pareto_indices
from pumo.model import Journey, Leg, validate_journey


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
