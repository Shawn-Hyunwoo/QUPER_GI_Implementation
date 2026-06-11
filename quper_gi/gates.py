"""Parametrized gates used by the QuPer ansatz.

The intended endpoint semantics are:

    param_cx(0)  -> identity
    param_cx(pi) -> CNOT

    param_swap(0)  -> identity
    param_swap(pi) -> SWAP

These gates are used as continuous switches during variational optimization.
"""

from __future__ import annotations

import pennylane as qml


def param_cx(theta, control: int, target: int) -> None:
    """Apply a parametrized-CX switch.

    Implementation (paper Appendix B, Fig. 19):
        PhaseShift(theta / 2) on the control wire
        CRX(theta) on [control, target]

    The phase gate anticipates the relative phase introduced by CRX so that
    theta=0 gives the identity and theta=pi gives exactly CNOT (no residual
    phase): crx(pi) . p(pi/2) = cx.
    """
    qml.PhaseShift(theta / 2.0, wires=control)
    qml.CRX(theta, wires=[control, target])


def param_swap(phi, wire_a: int, wire_b: int) -> None:
    """Apply a parametrized-SWAP switch.

    Decomposition (paper Fig. 4):
        CNOT(a, b)
        param_cx(phi, control=b, target=a)
        CNOT(a, b)
    """
    qml.CNOT(wires=[wire_a, wire_b])
    param_cx(phi, control=wire_b, target=wire_a)
    qml.CNOT(wires=[wire_a, wire_b])
