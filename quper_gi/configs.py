"""Configuration utilities for QuPer-GI experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Literal


AnsatzName = Literal["borel", "bruhat"]


@dataclass(frozen=True)
class GIConfig:
    """Experiment configuration.

    Defaults follow the paper (Sec. 6.2-6.3, Algorithm 1, Appendix C):
    Adam step size 0.4 for GI, regularizer weights 1/10, 15/100, 1/100,
    projection at every iteration with Hungarian + 50 random-order draws.

    Notes
    -----
    num_vertices:
        Graph vertex count N. Must be a power of two.

    ancillas:
        Number of ancilla pairs m. Total wires are 2*log2(N) + 2*m.

    diff_method:
        Use "best" for initial compatibility. For production, test "adjoint",
        "backprop", or framework-specific methods with the selected device.
    """

    num_vertices: int = 16
    ancillas: int = 0
    ansatz: AnsatzName = "borel"

    device: str = "default.qubit"
    diff_method: str = "best"

    graph_p: float = 0.5
    seed: int = 0
    perm_source: str = "uniform"  # "uniform" | "borel" | "bruhat" (paper Fig. 18)

    steps: int = 200
    lr: float = 0.4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8

    lambda_stochastic: float = 0.1
    lambda_entropy: float = 0.15
    lambda_orthogonal: float = 0.01

    projection_interval: int = 1
    num_random_projections: int = 50


def is_power_of_two(n: int) -> bool:
    """Return True iff n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def validate_config(cfg: GIConfig) -> tuple[int, int]:
    """Validate config and return (q, total_wires)."""
    if not is_power_of_two(cfg.num_vertices):
        raise ValueError("num_vertices must be a power of two.")

    if cfg.ancillas < 0:
        raise ValueError("ancillas must be non-negative.")

    if cfg.ansatz not in {"borel", "bruhat"}:
        raise ValueError("ansatz must be either 'borel' or 'bruhat'.")

    if cfg.projection_interval <= 0:
        raise ValueError("projection_interval must be positive.")

    if cfg.perm_source not in {"uniform", "borel", "bruhat"}:
        raise ValueError("perm_source must be 'uniform', 'borel', or 'bruhat'.")

    q = int(log2(cfg.num_vertices))
    total_wires = 2 * q + 2 * cfg.ancillas
    return q, total_wires


def summarize_config(cfg: GIConfig) -> str:
    """Return a concise human-readable config summary."""
    q, total_wires = validate_config(cfg)
    q_u = q + cfg.ancillas

    return (
        f"N={cfg.num_vertices}, q={q}, m={cfg.ancillas}, "
        f"qU={q_u}, wires={total_wires}, ansatz={cfg.ansatz}, "
        f"device={cfg.device}, diff_method={cfg.diff_method}"
    )
