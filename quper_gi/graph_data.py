"""Graph generation and hidden-permutation GI / SGI instances."""

from __future__ import annotations

import numpy as np


def generate_er_graph(num_vertices: int, edge_prob: float, seed: int) -> np.ndarray:
    """Generate an undirected Erdos-Renyi graph adjacency matrix.

    The graph has no self-loops.
    """
    if not 0.0 <= edge_prob <= 1.0:
        raise ValueError("edge_prob must be in [0, 1].")

    rng = np.random.default_rng(seed)
    upper = rng.random((num_vertices, num_vertices)) < edge_prob
    upper = np.triu(upper, k=1).astype(np.float64)
    adjacency = upper + upper.T
    return adjacency


def permutation_vector_to_matrix(perm: np.ndarray) -> np.ndarray:
    """Convert a permutation vector to a permutation matrix.

    Convention:
        P[i, perm[i]] = 1.

    With this convention, P @ x permutes x according to row assignments.
    Keep this convention consistent throughout the repository.
    """
    perm = np.asarray(perm, dtype=np.int64)
    n = int(perm.size)

    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[np.arange(n), perm] = 1.0
    return matrix


def random_permutation_matrix(num_vertices: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (P, perm) for a random hidden permutation."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_vertices)
    return permutation_vector_to_matrix(perm), perm


def span_permutation_matrix(
    num_vertices: int,
    seed: int,
    ansatz: str = "bruhat",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a hidden permutation from the span of the m = 0 circuit.

    With m = 0 and every parameter in {0, pi}, the circuit output P_hat is
    exactly a permutation matrix in the Borel/Bruhat span (paper Prop. 7).
    Instances built this way are guaranteed reachable by the ansatz and match
    the experiment of paper Fig. 18. Random permutations of n symbols are
    mostly OUTSIDE the span (e.g. |Bruhat span| = 322,560 vs 16! for n = 16),
    so uniform instances are usually not exactly solvable by the heuristic.
    """
    from pennylane import numpy as pnp

    from .dsm_circuit import QuPerDSM

    model = QuPerDSM(num_vertices=num_vertices, ancillas=0, ansatz=ansatz)
    rng = np.random.default_rng(seed)
    theta = pnp.array(
        rng.choice([0.0, np.pi], size=model.num_params), requires_grad=False
    )

    p_mat = np.rint(np.asarray(model(theta), dtype=np.float64))

    n = num_vertices
    if not (
        np.allclose(p_mat.sum(axis=0), np.ones(n))
        and np.allclose(p_mat.sum(axis=1), np.ones(n))
    ):
        raise RuntimeError("Span sampling did not produce a permutation matrix.")

    perm = np.argmax(p_mat, axis=1)
    return p_mat, perm


def make_isomorphic_pair(
    num_vertices: int,
    edge_prob: float,
    seed: int,
    perm_source: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a known-isomorphic GI instance.

    Parameters
    ----------
    perm_source:
        "uniform" draws the hidden permutation uniformly from S_n;
        "borel"/"bruhat" draw it from the corresponding m = 0 circuit span
        (guaranteed reachable; paper Fig. 18 setting).

    Returns
    -------
    A:
        Original adjacency matrix.

    B:
        Relabeled graph adjacency matrix.

    P_star:
        Hidden permutation matrix satisfying A = P_star @ B @ P_star.T.

    perm:
        Hidden permutation vector.

    Construction
    ------------
    B = P_star.T @ A @ P_star
    so A = P_star @ B @ P_star.T.
    """
    a_mat = generate_er_graph(num_vertices, edge_prob, seed)

    if perm_source == "uniform":
        p_star, perm = random_permutation_matrix(num_vertices, seed + 1)
    elif perm_source in {"borel", "bruhat"}:
        p_star, perm = span_permutation_matrix(num_vertices, seed + 1, ansatz=perm_source)
    else:
        raise ValueError("perm_source must be 'uniform', 'borel', or 'bruhat'.")

    b_mat = p_star.T @ a_mat @ p_star

    return a_mat, b_mat, p_star, perm


def verify_isomorphic_pair(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    p_star: np.ndarray,
    atol: float = 1.0e-9,
) -> None:
    """Raise if A != P_star @ B @ P_star.T."""
    residual = a_mat - p_star @ b_mat @ p_star.T
    error = float(np.sum(residual * residual))

    if error > atol:
        raise ValueError(f"Generated instance failed verification. error={error}")


# ---------------------------------------------------------------------------
# Subgraph isomorphism (SGI)
#
# Convention: the pattern H occupies the FIRST `pattern_size` vertices of the
# host graph; the remaining vertices are padding. The hidden permutation P*
# scrambles the host into the target B exactly as in the GI case, so the same
# P-convention and B = P*.T @ host @ P* relation hold. Only the loss is masked
# to the pattern's vertex pairs.
# ---------------------------------------------------------------------------


def subgraph_mask(
    host: np.ndarray,
    pattern_size: int,
    induced: bool = False,
) -> np.ndarray:
    """Build the constraint mask over the pattern's vertex pairs.

    The mask selects which entries of ``host - P B P^T`` are penalized:

    - ``induced=False`` (monomorphism): only the pattern's EDGES are constrained
      (edge-preserving map; non-edges of H are free to map onto edges of B).
    - ``induced=True`` (induced subgraph): ALL pattern vertex pairs are
      constrained, so non-edges of H must map onto non-edges of B too.

    The diagonal is always excluded (no self-loops).
    """
    n = host.shape[0]
    if not 1 <= pattern_size <= n:
        raise ValueError("pattern_size must be in [1, num_vertices].")

    mask = np.zeros((n, n), dtype=np.float64)
    mask[:pattern_size, :pattern_size] = 1.0
    np.fill_diagonal(mask, 0.0)

    if not induced:
        mask = mask * (host > 0.0)

    return mask


def make_subgraph_instance(
    num_vertices: int,
    pattern_size: int,
    edge_prob: float,
    seed: int,
    perm_source: str = "uniform",
    induced: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a known-solvable SGI instance with a planted embedding.

    The pattern H is the induced subgraph on the first ``pattern_size`` vertices
    of a random host graph, so H is by construction a subgraph of the host (and
    an induced subgraph). The host is scrambled by a hidden permutation P* into
    the target B, giving a guaranteed-correct embedding P*.

    Returns
    -------
    host:
        Host adjacency matrix; serves as the masked target (its top-left
        ``pattern_size`` block is the pattern H).

    B:
        Target graph, B = P*.T @ host @ P*.

    P_star:
        Hidden permutation matrix; the planted solution.

    perm:
        Hidden permutation vector.

    mask:
        Constraint mask from :func:`subgraph_mask` (monomorphism or induced).
    """
    if not 1 <= pattern_size <= num_vertices:
        raise ValueError("pattern_size must be in [1, num_vertices].")

    host = generate_er_graph(num_vertices, edge_prob, seed)

    if perm_source == "uniform":
        p_star, perm = random_permutation_matrix(num_vertices, seed + 1)
    elif perm_source in {"borel", "bruhat"}:
        p_star, perm = span_permutation_matrix(num_vertices, seed + 1, ansatz=perm_source)
    else:
        raise ValueError("perm_source must be 'uniform', 'borel', or 'bruhat'.")

    b_mat = p_star.T @ host @ p_star
    mask = subgraph_mask(host, pattern_size, induced=induced)

    return host, b_mat, p_star, perm, mask


def verify_subgraph_instance(
    host: np.ndarray,
    b_mat: np.ndarray,
    p_star: np.ndarray,
    mask: np.ndarray,
    atol: float = 1.0e-9,
) -> None:
    """Raise if the planted P* does not satisfy the masked constraints."""
    residual = (host - p_star @ b_mat @ p_star.T) * mask
    error = float(np.sum(residual * residual))

    if error > atol:
        raise ValueError(f"Generated SGI instance failed verification. error={error}")
