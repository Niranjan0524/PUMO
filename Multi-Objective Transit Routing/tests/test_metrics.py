import numpy as np
from pumo.bench.metrics import hypervolume, hypervolume_mc


def test_exact_hypervolume_two_dimensions():
    assert abs(hypervolume(np.array([[1,2],[2,1]]),[3,3])-3.0)<1e-12


def test_exact_hypervolume_three_dimensions():
    assert abs(hypervolume(np.array([[1,1,1]]),[3,3,3])-8.0)<1e-12


def test_monte_carlo_hypervolume_matches_exact_on_toy():
    points=np.array([[.2,.6],[.6,.2]])
    exact=hypervolume(points,[1,1])
    estimate=hypervolume_mc(points,[1,1],samples=200_000,seed=3)
    assert abs(exact-estimate)<.01
