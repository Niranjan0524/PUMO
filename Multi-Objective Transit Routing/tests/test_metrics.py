import numpy as np
from pumo.bench.metrics import hypervolume, hypervolume_mc, result_metrics


def test_exact_hypervolume_two_dimensions():
    assert abs(hypervolume(np.array([[1,2],[2,1]]),[3,3])-3.0)<1e-12


def test_exact_hypervolume_three_dimensions():
    assert abs(hypervolume(np.array([[1,1,1]]),[3,3,3])-8.0)<1e-12


def test_monte_carlo_hypervolume_matches_exact_on_toy():
    points=np.array([[.2,.6],[.6,.2]])
    exact=hypervolume(points,[1,1])
    estimate=hypervolume_mc(points,[1,1],samples=200_000,seed=3)
    assert abs(exact-estimate)<.01


def test_recall_counts_reference_journeys_missing_from_returned_set():
    reference=[{"T":1.0,"C":5.0},{"T":5.0,"C":1.0}]
    returned=[reference[0]]
    metrics=result_metrics(returned,reference,("T","C"),global_scales={"T":10,"C":10})
    assert metrics["recall"]==0.5
