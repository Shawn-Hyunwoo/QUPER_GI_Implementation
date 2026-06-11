"""Subgraph isomorphism (SGI) building blocks.

Checks that the masked-loss formulation correctly generalizes GI and that
planted SGI instances are solved exactly by their planted embedding P*.
"""

import numpy as np

from quper_gi.graph_data import (
    generate_er_graph,
    make_subgraph_instance,
    random_permutation_matrix,
    subgraph_mask,
    verify_subgraph_instance,
)
from quper_gi.losses import gi_frobenius_loss, masked_frobenius_loss
from quper_gi.metrics import is_sgi_success, masked_mismatch, subgraph_violations


def test_gi_is_masked_special_case():
    """Masked loss with an all-ones off-diagonal mask equals the GI loss."""
    a_mat = generate_er_graph(8, 0.5, seed=0)
    p_mat, _ = random_permutation_matrix(8, seed=1)
    b_mat = p_mat.T @ a_mat @ p_mat

    full_mask = np.ones((8, 8))
    np.fill_diagonal(full_mask, 0.0)

    gi = float(gi_frobenius_loss(p_mat, a_mat, b_mat))
    masked = float(masked_frobenius_loss(p_mat, a_mat, b_mat, full_mask))
    # GI loss includes the (zero) diagonal, so excluding it cannot change a
    # match; both are 0 at the planted solution and equal in general off-diag.
    assert np.isclose(gi, masked)


def test_monomorphism_mask_is_subset_of_induced():
    host = generate_er_graph(8, 0.5, seed=2)
    mono = subgraph_mask(host, pattern_size=4, induced=False)
    induced = subgraph_mask(host, pattern_size=4, induced=True)

    # Monomorphism constrains only edges; induced constrains all pattern pairs.
    assert np.all(mono <= induced)
    # Both vanish outside the pattern block and on the diagonal.
    assert induced[4:, :].sum() == 0 and induced[:, 4:].sum() == 0
    assert np.all(np.diag(induced) == 0)


def test_planted_monomorphism_is_solved():
    host, b_mat, p_star, _, mask = make_subgraph_instance(
        num_vertices=8, pattern_size=4, edge_prob=0.5, seed=3, induced=False
    )
    verify_subgraph_instance(host, b_mat, p_star, mask)
    assert masked_mismatch(host, b_mat, p_star, mask) == 0.0
    assert subgraph_violations(host, b_mat, p_star, mask) == 0
    assert is_sgi_success(host, b_mat, p_star, mask)


def test_planted_induced_is_solved():
    host, b_mat, p_star, _, mask = make_subgraph_instance(
        num_vertices=8, pattern_size=5, edge_prob=0.5, seed=4, induced=True
    )
    verify_subgraph_instance(host, b_mat, p_star, mask)
    assert is_sgi_success(host, b_mat, p_star, mask)


def test_wrong_permutation_violates_constraints():
    host, b_mat, p_star, _, mask = make_subgraph_instance(
        num_vertices=8, pattern_size=4, edge_prob=0.5, seed=5, induced=True
    )
    wrong, _ = random_permutation_matrix(8, seed=999)
    # A random permutation should break at least one pattern constraint.
    assert subgraph_violations(host, b_mat, wrong, mask) > 0
    assert not is_sgi_success(host, b_mat, wrong, mask)
