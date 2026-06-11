"""Losses and regularizers for QuPer-GI."""

from __future__ import annotations

import pennylane as qml


def gi_frobenius_loss(p_mat, a_mat, b_mat):
    """Continuous GI loss: ||A - P B P^T||_F^2."""
    residual = a_mat - p_mat @ b_mat @ qml.math.transpose(p_mat)
    return qml.math.sum(residual * residual)


def masked_frobenius_loss(p_mat, target_mat, b_mat, mask):
    """Masked SGI loss: sum_{ij} mask_ij * (target_ij - (P B P^T)_ij)^2.

    Generalizes the GI loss: with ``mask`` an all-ones off-diagonal matrix and
    ``target_mat = A`` this equals ``gi_frobenius_loss``. For subgraph
    isomorphism, ``mask`` restricts the penalty to the pattern's vertex pairs
    (edges only for monomorphism, all pairs for the induced variant); see
    :func:`quper_gi.graph_data.subgraph_mask`.
    """
    mapped = p_mat @ b_mat @ qml.math.transpose(p_mat)
    residual = (target_mat - mapped) * mask
    return qml.math.sum(residual * residual)


def stochastic_regularizer(p_mat):
    """Penalize row/column sum deviations from 1.

    For an ideal DSM, this should already be near zero. Keeping this optional is useful
    when numerical errors or modified circuits are introduced.
    """
    row_sums = qml.math.sum(p_mat, axis=1)
    col_sums = qml.math.sum(p_mat, axis=0)

    row_error = qml.math.sum((row_sums - 1.0) ** 2)
    col_error = qml.math.sum((col_sums - 1.0) ** 2)

    return row_error + col_error


def entropy_regularizer(p_mat, eps: float = 1.0e-12):
    """Entropy-like sharpening regularizer.

    Minimizing this pushes the matrix toward lower entropy / sharper entries.
    """
    return -qml.math.sum(p_mat * qml.math.log(p_mat + eps))


def orthogonality_regularizer(p_mat):
    """Penalize deviation from P^T P = I."""
    n = p_mat.shape[0]
    identity = qml.math.eye(n)
    gram = qml.math.transpose(p_mat) @ p_mat
    residual = gram - identity
    return qml.math.sum(residual * residual)


def total_loss(theta, model, a_mat, b_mat, cfg, mask=None, target_mat=None):
    """Regularized differentiable objective (paper Sec. 6.2, ell_R).

    Without ``mask`` this is the GI objective on ``a_mat``. With ``mask`` it is
    the masked SGI objective on ``target_mat`` (defaulting to ``a_mat``, since
    the host doubles as the target in :func:`make_subgraph_instance`). The
    regularizers act on the full DSM in both cases.
    """
    p_hat = model(theta)

    if mask is None:
        base = gi_frobenius_loss(p_hat, a_mat, b_mat)
    else:
        target = a_mat if target_mat is None else target_mat
        base = masked_frobenius_loss(p_hat, target, b_mat, mask)

    reg = 0.0
    if cfg.lambda_stochastic != 0.0:
        reg = reg + cfg.lambda_stochastic * stochastic_regularizer(p_hat)

    if cfg.lambda_entropy != 0.0:
        reg = reg + cfg.lambda_entropy * entropy_regularizer(p_hat)

    if cfg.lambda_orthogonal != 0.0:
        reg = reg + cfg.lambda_orthogonal * orthogonality_regularizer(p_hat)

    return base + reg
