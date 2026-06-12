"""QuPer circuit that outputs a doubly-stochastic matrix."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

import pennylane as qml

from .ansatz import apply_ansatz, get_num_params


@dataclass(frozen=True)
class WireLayout:
    """Wire layout for the QuPer DSM circuit."""

    anc_ref: list[int]
    row: list[int]
    anc_u: list[int]
    col: list[int]

    @property
    def u_wires(self) -> list[int]:
        """Wires on which U(theta) acts."""
        return self.anc_u + self.col

    @property
    def matrix_wires(self) -> list[int]:
        """Wires measured to get the N^2 matrix entries."""
        return self.row + self.col


def build_wire_layout(q: int, m: int) -> WireLayout:
    """Build the fixed QuPer-GI wire layout."""
    anc_ref = list(range(0, m))
    row = list(range(m, m + q))
    anc_u = list(range(m + q, 2 * m + q))
    col = list(range(2 * m + q, 2 * m + 2 * q))

    return WireLayout(
        anc_ref=anc_ref,
        row=row,
        anc_u=anc_u,
        col=col,
    )


class QuPerDSM:
    """DSM-producing QuPer circuit.

    Parameters
    ----------
    num_vertices:
        Graph vertex count N = 2^q.

    ancillas:
        Number of ancilla pairs m.

    ansatz:
        "borel" or "bruhat".

    device_name:
        Example: "default.qubit", "lightning.qubit", "lightning.gpu".

    diff_method:
        Use "best" initially. Benchmark other methods later.
    """

    def __init__(
        self,
        num_vertices: int,
        ancillas: int,
        ansatz: str,
        device_name: str = "default.qubit",
        diff_method: str = "best",
        visual_barriers: bool = False,
    ) -> None:
        q_float = log2(num_vertices)
        if int(q_float) != q_float:
            raise ValueError("num_vertices must be a power of two.")

        self.num_vertices = int(num_vertices)
        self.q = int(q_float)
        self.ancillas = int(ancillas)
        self.ansatz = ansatz

        self.visual_barriers = bool(visual_barriers)
        self.total_wires = 2 * self.q + 2 * self.ancillas
        self.layout = build_wire_layout(self.q, self.ancillas)
        self.q_u = len(self.layout.u_wires)
        self.num_params = get_num_params(ansatz, self.q_u)

        self.dev = qml.device(device_name, wires=self.total_wires)

        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="autograd",
            diff_method=diff_method,
        )

    def _prepare_bell_pairs(self) -> None:
        """Prepare maximally entangled pairs between reference and active registers."""
        for ref_wire, u_wire in zip(self.layout.anc_ref, self.layout.anc_u):
            qml.Hadamard(wires=ref_wire)
            qml.CNOT(wires=[ref_wire, u_wire])

        for row_wire, col_wire in zip(self.layout.row, self.layout.col):
            qml.Hadamard(wires=row_wire)
            qml.CNOT(wires=[row_wire, col_wire])

    def _circuit(self, theta):
        """Internal QNode circuit returning probabilities over row+col wires."""
        self._prepare_bell_pairs()
        if self.visual_barriers:
            qml.Barrier(wires=range(self.total_wires), only_visual=True)
        apply_ansatz(
            theta, self.layout.u_wires, self.ansatz, barriers=self.visual_barriers
        )
        return qml.probs(wires=self.layout.matrix_wires)

    def probs_to_matrix(self, probs):
        """Convert the N^2 probability vector to the N x N doubly-stochastic matrix.

        Paper Eq. (3): P_hat[i, j] = N * Tr(rho |ij><ij|). The measured
        probabilities sum to 1 over N^2 outcomes, so the factor N makes every
        row and column of P_hat sum to 1.
        """
        matrix = qml.math.reshape(probs, (self.num_vertices, self.num_vertices))
        return self.num_vertices * matrix

    def __call__(self, theta):
        """Return P_hat(theta)."""
        probs = self.qnode(theta)
        return self.probs_to_matrix(probs)

    def summary(self) -> str:
        """Return a concise model summary."""
        return (
            f"QuPerDSM(N={self.num_vertices}, q={self.q}, m={self.ancillas}, "
            f"qU={self.q_u}, wires={self.total_wires}, ansatz={self.ansatz}, "
            f"params={self.num_params})"
        )
