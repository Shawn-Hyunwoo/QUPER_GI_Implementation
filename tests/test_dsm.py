"""DSM property tests."""

import numpy as np
from pennylane import numpy as pnp

from quper_gi.dsm_circuit import QuPerDSM


def _assert_doubly_stochastic(p_hat: np.ndarray, n: int) -> None:
    assert p_hat.shape == (n, n)
    assert np.allclose(p_hat.sum(axis=0), np.ones(n), atol=1e-6)
    assert np.allclose(p_hat.sum(axis=1), np.ones(n), atol=1e-6)
    assert np.all(p_hat >= -1e-9)


def test_dsm_identity_at_theta_zero():
    """At theta = 0, U = I, so P_hat = I (paper: P_hat = U . conj(U) for m = 0)."""
    model = QuPerDSM(num_vertices=4, ancillas=0, ansatz="borel")
    theta = pnp.zeros(model.num_params, requires_grad=True)
    p_hat = np.asarray(model(theta))
    assert np.allclose(p_hat, np.eye(4), atol=1e-8)


def test_dsm_random_theta_borel():
    model = QuPerDSM(num_vertices=4, ancillas=0, ansatz="borel")
    rng = np.random.default_rng(0)
    theta = pnp.array(rng.uniform(0, np.pi, size=model.num_params), requires_grad=True)
    p_hat = np.asarray(model(theta))
    _assert_doubly_stochastic(p_hat, 4)


def test_dsm_random_theta_bruhat_with_ancilla():
    model = QuPerDSM(num_vertices=4, ancillas=1, ansatz="bruhat")
    assert model.total_wires == 6
    rng = np.random.default_rng(1)
    theta = pnp.array(rng.uniform(0, np.pi, size=model.num_params), requires_grad=True)
    p_hat = np.asarray(model(theta))
    _assert_doubly_stochastic(p_hat, 4)


def test_dsm_permutation_endpoints_give_permutation():
    """With all parameters in {0, pi} and m = 0, P_hat must be a hard permutation."""
    model = QuPerDSM(num_vertices=4, ancillas=0, ansatz="bruhat")
    rng = np.random.default_rng(2)
    theta = pnp.array(
        rng.choice([0.0, np.pi], size=model.num_params), requires_grad=True
    )
    p_hat = np.asarray(model(theta))
    _assert_doubly_stochastic(p_hat, 4)
    assert np.all(
        np.isclose(p_hat, 0.0, atol=1e-8) | np.isclose(p_hat, 1.0, atol=1e-8)
    )
