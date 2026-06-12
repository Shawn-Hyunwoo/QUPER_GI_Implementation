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


def plot_sgi_matching(
    host: np.ndarray,
    b_mat: np.ndarray,
    p_best: np.ndarray,
    mask: np.ndarray,
    pattern_size: int,
    path: Path,
    max_vertices: int = 32,
) -> bool:
    """Show where the pattern embeds in the target graph.

    Left: the pattern H (first `pattern_size` host vertices). Right: the full
    target B; vertices receiving the pattern share the pattern vertex's color,
    other vertices are gray. Image edges of satisfied pattern constraints are
    drawn thick; violated constraints are dashed red. Returns False if N is
    too large to draw.
    """
    n = host.shape[0]
    k = pattern_size
    if n > max_vertices:
        print(f"[visualize] skipping SGI plot: N={n} > {max_vertices}")
        return False

    sigma = np.argmax(np.asarray(p_best), axis=1).astype(int)

    pattern = nx.from_numpy_array(host[:k, :k])
    graph_b = nx.from_numpy_array(b_mat)
    pos_h = nx.spring_layout(pattern, seed=7)
    pos_b = nx.spring_layout(graph_b, seed=7)

    cmap = plt.get_cmap("turbo")
    pat_colors = [cmap(i / max(1, k - 1)) for i in range(k)]
    image_of = {i: int(sigma[i]) for i in range(k)}

    ok_edges, bad_edges = [], []
    for i in range(k):
        for j in range(i + 1, k):
            if mask[i, j] <= 0:
                continue
            bi, bj = image_of[i], image_of[j]
            satisfied = np.isclose(b_mat[bi, bj], host[i, j])
            (ok_edges if satisfied else bad_edges).append((bi, bj))

    fig, (ax_h, ax_b) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.6]}
    )

    nx.draw_networkx_edges(pattern, pos_h, ax=ax_h, width=1.8)
    nx.draw_networkx_nodes(
        pattern, pos_h, ax=ax_h, node_color=pat_colors, node_size=500, edgecolors="black"
    )
    nx.draw_networkx_labels(pattern, pos_h, ax=ax_h, font_size=10)
    ax_h.set_title(f"pattern H ({k} vertices)")
    ax_h.set_axis_off()

    node_colors = [
        pat_colors[sigma.tolist().index(v)] if v in image_of.values() else "lightgray"
        for v in range(n)
    ]
    node_sizes = [500 if v in image_of.values() else 220 for v in range(n)]
    nx.draw_networkx_edges(graph_b, pos_b, ax=ax_b, alpha=0.3)
    edges_present = [(u, v) for (u, v) in ok_edges if b_mat[u, v] > 0]
    if edges_present:
        nx.draw_networkx_edges(
            graph_b, pos_b, edgelist=edges_present, ax=ax_b, width=2.6
        )
    if bad_edges:
        nx.draw_networkx_edges(
            graph_b, pos_b, edgelist=bad_edges, ax=ax_b,
            edge_color="red", style="dashed", width=2.2,
        )
    nx.draw_networkx_nodes(
        graph_b, pos_b, ax=ax_b, node_color=node_colors, node_size=node_sizes,
        edgecolors="black",
    )
    labels = {v: str(v) for v in range(n)}
    for i, bv in image_of.items():
        labels[bv] = f"{bv}\n(h{i})"
    nx.draw_networkx_labels(graph_b, pos_b, ax=ax_b, labels=labels, font_size=8)

    title = "target B — pattern image highlighted (same colors)"
    title += (
        " — embedding satisfied"
        if not bad_edges
        else f" — {len(bad_edges)} violated constraints (red)"
    )
    ax_b.set_title(title)
    ax_b.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] SGI matching plot saved to: {path}")
    return True


def plot_sgi_heatmaps(
    host: np.ndarray,
    b_mat: np.ndarray,
    p_best: np.ndarray,
    mask: np.ndarray,
    pattern_size: int,
    path: Path,
) -> None:
    """SGI heatmap (any N).

    Panels: (1) the target B with the selected rows/columns marked — the k
    vertices sigma(0..k-1) the permutation assigned to the pattern slots;
    (2) the pattern H (k x k); (3) the k x k submatrix of B extracted at those
    vertices, slot-ordered, i.e. B[sigma_i, sigma_j] = (P B P^T)[i, j];
    (4) the graded difference mask * |H - extracted|. Under monomorphism the
    extracted block may contain EXTRA edges (superset of H's edges) — only
    H's edge positions are graded; under induced the two blocks must match.
    """
    k = pattern_size
    sigma = np.argmax(np.asarray(p_best), axis=1).astype(int)[:k]

    pattern = host[:k, :k]
    raw_crop = b_mat[np.ix_(np.sort(sigma), np.sort(sigma))]
    extracted = b_mat[np.ix_(sigma, sigma)]
    block_mask = mask[:k, :k]
    residual = np.abs(pattern - extracted) * block_mask

    def edges(m):
        return int(m.sum() // 2)

    fig, axes = plt.subplots(
        1, 5, figsize=(19, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1, 1, 1]}
    )

    im = axes[0].imshow(b_mat, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    for v in sigma:
        axes[0].axhline(v, color="red", lw=0.8, alpha=0.45)
        axes[0].axvline(v, color="red", lw=0.8, alpha=0.45)
    axes[0].set_title(f"target B — selected {k} rows/cols (red)", fontsize=10)

    panels = (
        (axes[1], raw_crop, f"crop as-is, ascending ({edges(raw_crop)} edges)"),
        (axes[2], extracted, f"reordered to slots ({edges(extracted)} edges)"),
        (axes[3], pattern, f"pattern H ({edges(pattern)} edges)"),
        (axes[4], residual, r"mask $\odot$ |H $-$ reordered|"),
    )
    for ax, mat, title in panels:
        im = ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8)

    violations = int(residual.sum() // 2)
    fig.suptitle(
        "embedding satisfied" if violations == 0
        else f"{violations} violated constraints",
        y=1.02,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] SGI heatmaps saved to: {path}")


def plot_fom_panel(history: list[dict], scalars: dict, path: Path) -> None:
    """Aggregate FOM panel: grad norm, sharpness, search diversity, scalars.

    Panels use whatever keys exist in the history records (older histories
    without instrumentation simply produce empty panels).
    """
    def series(key):
        pts = [(r["step"], r[key]) for r in history if key in r and r[key] == r[key]]
        return ([p[0] for p in pts], [p[1] for p in pts])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    s, v = series("grad_norm")
    axes[0, 0].plot(s, v, lw=1.2)
    axes[0, 0].set_title("gradient norm")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("step")

    s1, v1 = series("p_hat_entropy")
    s2, v2 = series("projection_residual")
    axes[0, 1].plot(s1, v1, lw=1.2, label="P_hat entropy")
    axes[0, 1].plot(s2, v2, lw=1.2, label="||P_hat - P_proj||_F")
    axes[0, 1].set_title("soft-matrix sharpness")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].legend()

    s, v = series("distinct_perms")
    axes[1, 0].plot(s, v, lw=1.2)
    axes[1, 0].set_title("distinct projected permutations (cumulative)")
    axes[1, 0].set_xlabel("step")

    axes[1, 1].set_axis_off()
    text = "\n".join(f"{k}: {v}" for k, v in scalars.items())
    axes[1, 1].text(0.02, 0.95, text, va="top", family="monospace", fontsize=10)
    axes[1, 1].set_title("scalar FOMs")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[visualize] FOM panel saved to: {path}")


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
