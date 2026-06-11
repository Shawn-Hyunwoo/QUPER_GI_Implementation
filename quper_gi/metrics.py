"""Metrics and validation helpers."""

from __future__ import annotations

import numpy as np


def permutation_validity(p_mat: np.ndarray, atol: float = 1.0e-9) -> bool:
    """Return True iff p_mat is a valid permutation matrix up to tolerance."""
    p_mat = np.asarray(p_mat)

    if p_mat.ndim != 2 or p_mat.shape[0] != p_mat.shape[1]:
        return False

    n = p_mat.shape[0]

    row_ok = np.allclose(p_mat.sum(axis=1), np.ones(n), atol=atol)
    col_ok = np.allclose(p_mat.sum(axis=0), np.ones(n), atol=atol)
    binary_ok = np.all((np.isclose(p_mat, 0.0, atol=atol)) | (np.isclose(p_mat, 1.0, atol=atol)))

    return bool(row_ok and col_ok and binary_ok)


def frobenius_mismatch(a_mat: np.ndarray, b_mat: np.ndarray, p_mat: np.ndarray) -> float:
    """Return ||A - P B P^T||_F^2."""
    residual = a_mat - p_mat @ b_mat @ p_mat.T
    return float(np.sum(residual * residual))


def edge_mismatch(a_mat: np.ndarray, b_mat: np.ndarray, p_mat: np.ndarray) -> int:
    """Return undirected edge mismatch count for binary adjacency matrices."""
    residual = np.abs(a_mat - p_mat @ b_mat @ p_mat.T)
    return int(np.sum(residual) // 2)


def is_success(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    p_mat: np.ndarray,
    tol: float = 1.0e-9,
) -> bool:
    """Success iff projected GI loss is zero within tolerance."""
    return frobenius_mismatch(a_mat, b_mat, p_mat) <= tol


def hidden_perm_overlap(p_mat: np.ndarray, p_star: np.ndarray) -> float:
    """Fraction of row assignments matching the hidden permutation.

    This is diagnostic only. Due to automorphisms, a different permutation can still be
    a valid GI certificate.
    """
    pred = np.argmax(p_mat, axis=1)
    truth = np.argmax(p_star, axis=1)
    return float(np.mean(pred == truth))


# --- Subgraph isomorphism (SGI) metrics ------------------------------------


def masked_mismatch(
    target: np.ndarray,
    b_mat: np.ndarray,
    p_mat: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return sum_{ij} mask_ij * (target_ij - (P B P^T)_ij)^2."""
    residual = (target - p_mat @ b_mat @ p_mat.T) * mask
    return float(np.sum(residual * residual))


def subgraph_violations(
    target: np.ndarray,
    b_mat: np.ndarray,
    p_mat: np.ndarray,
    mask: np.ndarray,
) -> int:
    """Count violated pattern constraints (undirected, so each pair once).

    A constraint at (i, j) with mask_ij = 1 is violated when target_ij differs
    from the mapped value (P B P^T)_ij.
    """
    residual = np.abs(target - p_mat @ b_mat @ p_mat.T) * mask
    return int(np.sum(residual > 0.5) // 2)


def is_sgi_success(
    target: np.ndarray,
    b_mat: np.ndarray,
    p_mat: np.ndarray,
    mask: np.ndarray,
    tol: float = 1.0e-9,
) -> bool:
    """Success iff every masked pattern constraint is satisfied."""
    return masked_mismatch(target, b_mat, p_mat, mask) <= tol
