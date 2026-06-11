"""Projection from a soft doubly-stochastic matrix to a hard permutation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def project_hungarian(p_hat: np.ndarray) -> np.ndarray:
    """Project P_hat to a permutation matrix by maximizing <P, P_hat>."""
    p_hat = np.asarray(p_hat)
    n = p_hat.shape[0]

    row_ind, col_ind = linear_sum_assignment(-p_hat)

    p_mat = np.zeros((n, n), dtype=np.float64)
    p_mat[row_ind, col_ind] = 1.0

    return p_mat


def project_random_order(p_hat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random-order projection (paper Appendix C).

    Start from a vector with entries v_i = 2^i, so every coefficient is at
    least twice or half any other one; this prevents P_hat entries far from a
    permutation from unexpectedly changing the order of the returned vector.
    The vector is randomly permuted, y = P_hat @ v is computed, and the hard
    permutation P is the one matching the orderings: psi(P_hat v) = P psi(v).
    """
    p_hat = np.asarray(p_hat)
    n = p_hat.shape[0]

    base = 2.0 ** np.arange(n)
    x_vec = rng.permutation(base)
    y_vec = p_hat @ x_vec

    x_order = np.argsort(x_vec)
    y_order = np.argsort(y_vec)

    p_mat = np.zeros((n, n), dtype=np.float64)

    for rank in range(n):
        source = x_order[rank]
        target = y_order[rank]
        p_mat[target, source] = 1.0

    return p_mat


def projected_gi_loss_np(p_mat: np.ndarray, a_mat: np.ndarray, b_mat: np.ndarray) -> float:
    """Numpy projected GI loss."""
    residual = a_mat - p_mat @ b_mat @ p_mat.T
    return float(np.sum(residual * residual))


def project_best(
    p_hat: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    rng: np.random.Generator,
    num_random: int = 50,
    score_fn=None,
) -> tuple[np.ndarray, float, str]:
    """Try multiple projections and return the best by the true objective.

    Follows the paper: both the Hungarian assignment and `num_random`
    random-order projections are evaluated with the original (unregularized)
    objective, and the minimizer is returned. `score_fn(P) -> float` overrides
    the default GI loss (used for the masked SGI objective).
    """
    if score_fn is None:
        def score_fn(p_mat):
            return projected_gi_loss_np(p_mat, a_mat, b_mat)

    candidates: list[tuple[str, np.ndarray]] = []

    candidates.append(("hungarian", project_hungarian(p_hat)))

    for idx in range(num_random):
        candidates.append((f"random_order_{idx}", project_random_order(p_hat, rng)))

    best_name = ""
    best_value = float("inf")
    best_p = None

    for name, candidate in candidates:
        value = score_fn(candidate)

        if value < best_value:
            best_name = name
            best_value = value
            best_p = candidate

    if best_p is None:
        raise RuntimeError("Projection failed to produce candidates.")

    return best_p, best_value, best_name
