"""Visualization of GI matchings for small instances.

`plot_matching` draws graphs A and B side by side with the recovered mapping:
matched vertices share a color and a position (B is laid out so that vertex
sigma(i) sits at the same height as A's vertex i), with dashed connector lines
between matched pairs. If the matching is a true isomorphism, the two edge
drawings are mirror images. Mismatched edges are highlighted in red.

`plot_heatmaps` works at any size (including N = 256): adjacency matrices A,
P B P^T, and the absolute residual |A - P B P^T|.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _matching_from_p(p_best: np.ndarray) -> np.ndarray:
    """sigma[i] = B-vertex matched to A-vertex i (P[i, sigma(i)] = 1)."""
    return np.argmax(p_best, axis=1)


def plot_matching(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    p_best: np.ndarray,
    path: Path,
    max_vertices: int = 32,
) -> bool:
    """Draw the vertex matching between A and B. Returns False if N is too large."""
    n = a_mat.shape[0]
    if n > max_vertices:
        print(f"[visualize] skipping matching plot: N={n} > {max_vertices}")
        return False

    sigma = _matching_from_p(p_best)
    sigma_inv = np.argsort(sigma)

    graph_a = nx.from_numpy_array(a_mat)
    graph_b = nx.from_numpy_array(b_mat)

    # Layout A once; B vertex sigma(i) inherits A vertex i's position, mirrored
    # to the right panel. A correct matching then yields two identical drawings.
    pos_a = nx.spring_layout(graph_a, seed=7)
    span = 2.8
    pos_b = {int(sigma[i]): (span - pos_a[i][0], pos_a[i][1]) for i in range(n)}

    cmap = plt.get_cmap("turbo")
    colors_a = [cmap(i / max(1, n - 1)) for i in range(n)]
    colors_b = [colors_a[sigma_inv[v]] for v in range(n)]

    # Edge mismatch sets (residual is +1: edge in A unmatched; -1: extra edge image).
    residual = a_mat - p_best @ b_mat @ p_best.T
    bad_a = [(i, j) for i in range(n) for j in range(i + 1, n) if residual[i, j] > 0.5]
    image_b = p_best.T @ a_mat @ p_best  # A mapped into B labels
    residual_b = b_mat - image_b
    bad_b = [(i, j) for i in range(n) for j in range(i + 1, n) if residual_b[i, j] > 0.5]

    fig, ax = plt.subplots(figsize=(13, 6.5))

    nx.draw_networkx_edges(graph_a, pos_a, ax=ax, alpha=0.45)
    nx.draw_networkx_edges(graph_b, pos_b, ax=ax, alpha=0.45)
    if bad_a:
        nx.draw_networkx_edges(
            graph_a, pos_a, edgelist=bad_a, ax=ax, edge_color="red", width=2.2
        )
    if bad_b:
        nx.draw_networkx_edges(
            graph_b, pos_b, edgelist=bad_b, ax=ax, edge_color="red", width=2.2
        )

    nx.draw_networkx_nodes(
        graph_a, pos_a, ax=ax, node_color=colors_a, node_size=380, edgecolors="black"
    )
    nx.draw_networkx_nodes(
        graph_b, pos_b, ax=ax, node_color=colors_b, node_size=380, edgecolors="black"
    )
    nx.draw_networkx_labels(graph_a, pos_a, ax=ax, font_size=9)
    nx.draw_networkx_labels(graph_b, pos_b, ax=ax, font_size=9)

    for i in range(n):
        xa, ya = pos_a[i]
        xb, yb = pos_b[int(sigma[i])]
        ax.plot([xa, xb], [ya, yb], ls="--", lw=0.6, color="gray", alpha=0.5, zorder=0)

    num_bad = len(bad_a) + len(bad_b)
    title = "A (left)  <->  B (right), matched vertices share color and height"
    title += " — exact isomorphism" if num_bad == 0 else f" — {num_bad} mismatched edges (red)"
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] matching plot saved to: {path}")
    return True


def plot_heatmaps(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    p_best: np.ndarray,
    path: Path,
) -> None:
    """Adjacency heatmaps: A, P B P^T, and |A - P B P^T| (any N)."""
    mapped = p_best @ b_mat @ p_best.T
    residual = np.abs(a_mat - mapped)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, mat, title in (
        (axes[0], a_mat, "A"),
        (axes[1], mapped, r"$P\,B\,P^\top$"),
        (axes[2], residual, r"$|A - P B P^\top|$"),
    ):
        im = ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8)

    mismatches = int(residual.sum() // 2)
    fig.suptitle(
        "Exact match" if mismatches == 0 else f"{mismatches} mismatched edges",
        y=1.0,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] heatmaps saved to: {path}")


def plot_soft_matrix(p_soft: np.ndarray, p_best: np.ndarray, path: Path) -> None:
    """Soft DSM next to its hard projection."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))
    im0 = axes[0].imshow(p_soft, cmap="viridis", interpolation="nearest")
    axes[0].set_title(r"soft $\hat P_\theta$ (DSM)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    axes[1].imshow(p_best, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(r"projected $P_{best}$")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] soft-matrix plot saved to: {path}")
