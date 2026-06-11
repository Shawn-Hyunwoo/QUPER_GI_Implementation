"""Projection method tests."""

import numpy as np

from quper_gi.graph_data import make_isomorphic_pair, permutation_vector_to_matrix
from quper_gi.metrics import permutation_validity
from quper_gi.projection import (
    project_best,
    project_hungarian,
    project_random_order,
    projected_gi_loss_np,
)


def _random_permutation_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return permutation_vector_to_matrix(rng.permutation(n))


def test_hungarian_recovers_exact_permutation():
    p_star = _random_permutation_matrix(8, 0)
    assert np.allclose(project_hungarian(p_star), p_star)


def test_hungarian_recovers_noisy_permutation():
    p_star = _random_permutation_matrix(8, 1)
    rng = np.random.default_rng(2)
    noisy = p_star + 0.05 * rng.random((8, 8))
    assert np.allclose(project_hungarian(noisy), p_star)


def test_random_order_recovers_exact_permutation():
    p_star = _random_permutation_matrix(8, 3)
    rng = np.random.default_rng(4)
    for _ in range(5):
        projected = project_random_order(p_star, rng)
        assert permutation_validity(projected)
        assert np.allclose(projected, p_star)


def test_project_best_zero_loss_on_planted_instance():
    a_mat, b_mat, p_star, _ = make_isomorphic_pair(num_vertices=8, edge_prob=0.5, seed=5)
    rng = np.random.default_rng(6)
    p_best, value, name = project_best(p_star, a_mat, b_mat, rng, num_random=10)
    assert value == 0.0
    assert projected_gi_loss_np(p_best, a_mat, b_mat) == 0.0
    assert name
