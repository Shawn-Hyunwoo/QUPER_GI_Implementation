"""Endpoint tests for parametrized gates.

Paper Appendix B proves the endpoints are exact (no residual global phase):
    param_cx(0) = I,  param_cx(pi) = CNOT
    param_swap(0) = I, param_swap(pi) = SWAP
"""

import numpy as np
import pennylane as qml

from quper_gi.gates import param_cx, param_swap


def gate_matrix(fn, *args, **kwargs) -> np.ndarray:
    return qml.matrix(fn, wire_order=[0, 1])(*args, **kwargs)


def test_param_cx_zero_is_identity():
    mat = gate_matrix(param_cx, 0.0, control=0, target=1)
    assert np.allclose(mat, np.eye(4), atol=1e-10)


def test_param_cx_pi_is_cnot():
    mat = gate_matrix(param_cx, np.pi, control=0, target=1)
    expected = qml.matrix(qml.CNOT(wires=[0, 1]), wire_order=[0, 1])
    assert np.allclose(mat, expected, atol=1e-10)


def test_param_cx_pi_reversed_wires():
    mat = gate_matrix(param_cx, np.pi, control=1, target=0)
    expected = qml.matrix(qml.CNOT(wires=[1, 0]), wire_order=[0, 1])
    assert np.allclose(mat, expected, atol=1e-10)


def test_param_swap_zero_is_identity():
    mat = gate_matrix(param_swap, 0.0, wire_a=0, wire_b=1)
    assert np.allclose(mat, np.eye(4), atol=1e-10)


def test_param_swap_pi_is_swap():
    mat = gate_matrix(param_swap, np.pi, wire_a=0, wire_b=1)
    expected = qml.matrix(qml.SWAP(wires=[0, 1]), wire_order=[0, 1])
    assert np.allclose(mat, expected, atol=1e-10)
