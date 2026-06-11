"""Borel, Weyl, and Bruhat-style ansatz builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pennylane as qml

from .gates import param_cx, param_swap


@dataclass(frozen=True)
class AnsatzSlices:
    """Parameter slice metadata for an ansatz."""

    n_x: int
    n_b: int
    n_w: int
    total: int


def comb2(n: int) -> int:
    """Return C(n, 2)."""
    return n * (n - 1) // 2


def num_borel_params(q_u: int) -> int:
    """Parameter count for X layer + one Borel block."""
    return q_u + comb2(q_u)


def num_bruhat_params(q_u: int) -> int:
    """Parameter count for X layer + Borel + Weyl + Borel."""
    return q_u + 3 * comb2(q_u)


def get_num_params(ansatz: str, q_u: int) -> int:
    """Return parameter count for a named ansatz."""
    if ansatz == "borel":
        return num_borel_params(q_u)
    if ansatz == "bruhat":
        return num_bruhat_params(q_u)
    raise ValueError(f"Unknown ansatz: {ansatz}")


def x_layer(params, wires: Sequence[int]) -> None:
    """Apply one RX layer."""
    if len(params) != len(wires):
        raise ValueError("x_layer parameter length mismatch.")

    for theta, wire in zip(params, wires):
        qml.RX(theta, wires=wire)


def borel_block(params, wires: Sequence[int]) -> None:
    """Apply the Borel block.

    Convention (paper Sec. 4.3, Fig. 3)
    ------------------------------------
    Reading the longest word w* right to left, gates are applied target-major:
    for every target index i and control index j > i:

        param_cx(theta, control=wires[j], target=wires[i])
    """
    q_u = len(wires)
    expected = comb2(q_u)
    if len(params) != expected:
        raise ValueError(f"borel_block expected {expected} params, got {len(params)}.")

    idx = 0
    for target_idx in range(q_u - 1):
        for control_idx in range(target_idx + 1, q_u):
            param_cx(
                params[idx],
                control=wires[control_idx],
                target=wires[target_idx],
            )
            idx += 1


def weyl_block(params, wires: Sequence[int]) -> None:
    """Apply the Weyl block using the longest-word adjacent-swap pattern.

    Implements the word w_o (paper Sec. 4.4, Fig. 5): q_u - 1 blocks of
    cascading adjacent parametrized swaps with decreasing length.
    """
    q_u = len(wires)
    expected = comb2(q_u)
    if len(params) != expected:
        raise ValueError(f"weyl_block expected {expected} params, got {len(params)}.")

    idx = 0
    for block in range(q_u - 1):
        limit = q_u - 1 - block
        for i in range(limit):
            param_swap(params[idx], wires[i], wires[i + 1])
            idx += 1


def apply_borel_ansatz(theta, wires: Sequence[int]) -> None:
    """Apply X layer followed by one Borel block."""
    q_u = len(wires)
    n_x = q_u
    n_b = comb2(q_u)
    expected = n_x + n_b

    if len(theta) != expected:
        raise ValueError(f"Borel ansatz expected {expected} params, got {len(theta)}.")

    x_params = theta[:n_x]
    b_params = theta[n_x:n_x + n_b]

    x_layer(x_params, wires)
    borel_block(b_params, wires)


def apply_bruhat_ansatz(theta, wires: Sequence[int]) -> None:
    """Apply X-B-W-B Bruhat-style ansatz."""
    q_u = len(wires)
    n_x = q_u
    n_b = comb2(q_u)
    n_w = n_b
    expected = n_x + n_b + n_w + n_b

    if len(theta) != expected:
        raise ValueError(f"Bruhat ansatz expected {expected} params, got {len(theta)}.")

    offset = 0

    x_params = theta[offset:offset + n_x]
    offset += n_x

    b1_params = theta[offset:offset + n_b]
    offset += n_b

    w_params = theta[offset:offset + n_w]
    offset += n_w

    b2_params = theta[offset:offset + n_b]

    x_layer(x_params, wires)
    borel_block(b1_params, wires)
    weyl_block(w_params, wires)
    borel_block(b2_params, wires)


def apply_ansatz(theta, wires: Sequence[int], ansatz: str) -> None:
    """Dispatch to a named ansatz."""
    if ansatz == "borel":
        apply_borel_ansatz(theta, wires)
        return

    if ansatz == "bruhat":
        apply_bruhat_ansatz(theta, wires)
        return

    raise ValueError(f"Unknown ansatz: {ansatz}")


def wire_labels(q: int, m: int) -> list[tuple]:
    """Logical labels of the wires U(theta) acts on, in u_wires order.

    u_wires = anc_u + col, so the first m labels are ancilla roles and the
    remaining q labels are column-register roles. Labels are stable across
    different m, which makes parameter transfer between ancilla stages possible.
    """
    return [("anc", i) for i in range(m)] + [("col", j) for j in range(q)]


def param_keys(ansatz: str, q: int, m: int) -> list[tuple]:
    """Hashable identity of every parameter, in exact parameter order.

    Used to warm-start theta when the number of ancilla pairs is incremented
    (paper Algorithm 1, line 10): parameters whose key exists in both layouts
    are copied; the rest are padded.
    """
    labels = wire_labels(q, m)
    q_u = len(labels)

    keys: list[tuple] = [("x", lab) for lab in labels]

    def borel_keys(tag: str) -> list[tuple]:
        out: list[tuple] = []
        for t in range(q_u - 1):
            for c in range(t + 1, q_u):
                out.append((tag, labels[t], labels[c]))
        return out

    if ansatz == "borel":
        keys += borel_keys("b1")
    elif ansatz == "bruhat":
        keys += borel_keys("b1")
        for block in range(q_u - 1):
            limit = q_u - 1 - block
            for i in range(limit):
                keys.append(("w", block, labels[i], labels[i + 1]))
        keys += borel_keys("b2")
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")

    if len(keys) != get_num_params(ansatz, q_u):
        raise RuntimeError("param_keys is inconsistent with get_num_params.")

    return keys


def transfer_theta(
    theta_old: np.ndarray,
    ansatz: str,
    q: int,
    m_old: int,
    m_new: int,
    fill: float = np.pi / 8.0,
) -> np.ndarray:
    """Map a trained theta from m_old ancilla pairs onto the m_new layout.

    Matching parameters are copied; new parameters are initialized to `fill`
    (pi/8 in paper Algorithm 1).
    """
    old_keys = param_keys(ansatz, q, m_old)
    if len(theta_old) != len(old_keys):
        raise ValueError("theta_old length does not match the m_old layout.")

    lookup = dict(zip(old_keys, np.asarray(theta_old, dtype=np.float64)))
    new_keys = param_keys(ansatz, q, m_new)
    return np.array([lookup.get(key, fill) for key in new_keys], dtype=np.float64)
