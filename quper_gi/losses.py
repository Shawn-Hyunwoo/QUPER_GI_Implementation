"""Losses and regularizers for QuPer-GI."""

from __future__ import annotations

import pennylane as qml


def gi_frobenius_loss(p_mat, a_mat, b_mat):
    """Continuous GI loss: ||A - P B P^T||_F^2."""
    residual = a_mat - p_mat @ b_mat @ qml.math.transpose(p_mat)
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


def total_loss(theta, model, a_mat, b_mat, cfg):
    """Regularized differentiable objective (paper Sec. 6.2, ell_R)."""
    p_hat = model(theta)

    base = gi_frobenius_loss(p_hat, a_mat, b_mat)

    reg = 0.0
    if cfg.lambda_stochastic != 0.0:
        reg = reg + cfg.lambda_stochastic * stochastic_regularizer(p_hat)

    if cfg.lambda_entropy != 0.0:
        reg = reg + cfg.lambda_entropy * entropy_regularizer(p_hat)

    if cfg.lambda_orthogonal != 0.0:
        reg = reg + cfg.lambda_orthogonal * orthogonality_regularizer(p_hat)

    return base + reg
