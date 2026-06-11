"""Shared experiment harness: run, verify, organize, and plot.

Every experiment produces one self-contained output directory:

    config.json          experiment configuration
    result.json          best value/step/projection, success, timings, history
    verification.json    isomorphism verification of the returned permutation
    report.txt           human-readable summary
    adjacency_a.npy      instance graph A
    adjacency_b.npy      instance graph B
    p_star.npy           hidden permutation matrix
    best_permutation.npy best projected permutation found
    best_theta.npy       parameters of the best projection
    loss_curve.png       training loss and best projected loss vs step
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from quper_gi.configs import GIConfig, summarize_config, validate_config
from quper_gi.graph_data import make_isomorphic_pair, verify_isomorphic_pair
from quper_gi.metrics import (
    edge_mismatch,
    frobenius_mismatch,
    hidden_perm_overlap,
    is_success,
    permutation_validity,
)
from quper_gi.optimizer import train_quper, train_single_m


def verify_solution(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    p_best: np.ndarray,
    p_star: np.ndarray,
) -> dict:
    """Full isomorphism verification of a projected permutation."""
    sigma = np.argmax(p_best, axis=1)
    return {
        "valid_permutation": bool(permutation_validity(p_best)),
        "frobenius_mismatch": frobenius_mismatch(a_mat, b_mat, p_best),
        "edge_mismatch": edge_mismatch(a_mat, b_mat, p_best),
        "success": bool(is_success(a_mat, b_mat, p_best)),
        "hidden_overlap": hidden_perm_overlap(p_best, p_star),
        "sigma": sigma.tolist(),
    }


def write_report(
    path: Path,
    cfg: GIConfig,
    result: dict,
    verification: dict,
) -> None:
    n = cfg.num_vertices
    lines = [
        "QuPer-GI run report",
        "===================",
        f"Config: {summarize_config(cfg)}",
        f"Steps: {cfg.steps}, lr: {cfg.lr}, lambdas: "
        f"st={cfg.lambda_stochastic}, ent={cfg.lambda_entropy}, ort={cfg.lambda_orthogonal}",
        "",
        f"Best projected GI loss: {result['value']:.6g} "
        f"(step {result['step']}, via {result['projection']}"
        + (f", m={result['ancillas']}" if "ancillas" in result else "")
        + ")",
        f"Valid permutation: {verification['valid_permutation']}",
        f"Edge mismatch: {verification['edge_mismatch']} "
        f"(Frobenius: {verification['frobenius_mismatch']:.6g})",
        "",
        "SUCCESS: A == P B P^T — isomorphism certificate found."
        if verification["success"]
        else "NOT SOLVED: projected loss did not reach zero "
        "(heuristic — this does NOT prove non-isomorphism).",
        "",
        f"Hidden-permutation overlap: {verification['hidden_overlap']:.3f} "
        "(diagnostic only; automorphisms allow other valid certificates)",
        f"Elapsed: {result.get('elapsed_seconds', float('nan')):.1f} s",
    ]

    if n <= 32:
        sigma = verification["sigma"]
        lines += ["", "Recovered mapping sigma (A-vertex i -> B-vertex sigma(i)):"]
        lines += [f"  {i:3d} -> {s}" for i, s in enumerate(sigma)]

    if result.get("stages"):
        lines += ["", "Ancilla schedule stages:"]
        for stage in result["stages"]:
            lines.append(
                f"  m={stage['ancillas']}: best={stage['value']:.6g} "
                f"(step {stage['step']}, via {stage['projection']})"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_loss_curve(history: list[dict], path: Path) -> None:
    steps = [rec["step"] for rec in history]
    train = [rec["train_loss"] for rec in history]
    best = [rec["best_projected_loss"] for rec in history]

    proj_steps = [rec["step"] for rec in history if "projected_loss" in rec]
    proj = [rec["projected_loss"] for rec in history if "projected_loss" in rec]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, train, label="regularized train loss", lw=1.5)
    if proj_steps:
        ax.plot(proj_steps, proj, label="projected GI loss", lw=1.0, alpha=0.7)
    ax.plot(steps, best, label="best projected GI loss", lw=2.0)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("loss")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend()
    ax.set_title("QuPer-GI training")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(
    cfg: GIConfig,
    outdir: Path,
    use_ancilla_schedule: bool = False,
) -> dict:
    """Run one experiment end to end and write the organized output directory.

    Returns a dict with the training result, verification, and instance arrays.
    """
    validate_config(cfg)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Config:", summarize_config(cfg))

    a_mat, b_mat, p_star, _ = make_isomorphic_pair(
        num_vertices=cfg.num_vertices,
        edge_prob=cfg.graph_p,
        seed=cfg.seed,
        perm_source=cfg.perm_source,
    )
    verify_isomorphic_pair(a_mat, b_mat, p_star)

    t_start = time.perf_counter()
    if use_ancilla_schedule:
        result = train_quper(a_mat, b_mat, p_star, cfg)
    else:
        result = train_single_m(a_mat, b_mat, p_star, cfg)
    result["elapsed_seconds"] = time.perf_counter() - t_start

    if result.get("p") is None:
        raise RuntimeError("Training produced no projected permutation.")

    verification = verify_solution(a_mat, b_mat, result["p"], p_star)

    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    serializable_result = {
        "best_value": result["value"],
        "best_step": result["step"],
        "best_projection": result["projection"],
        "best_ancillas": result.get("ancillas"),
        "elapsed_seconds": result["elapsed_seconds"],
        "stages": result.get("stages"),
        "history": result.get("history", []),
    }
    with (outdir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_result, f, indent=2)

    with (outdir / "verification.json").open("w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2)

    np.save(outdir / "adjacency_a.npy", a_mat)
    np.save(outdir / "adjacency_b.npy", b_mat)
    np.save(outdir / "p_star.npy", p_star)
    np.save(outdir / "best_permutation.npy", result["p"])
    np.save(outdir / "best_theta.npy", result["theta"])

    write_report(outdir / "report.txt", cfg, result, verification)
    plot_loss_curve(result.get("history", []), outdir / "loss_curve.png")

    # Scalar FOMs (journal Sec 5.4) + instrumented-trajectory panel.
    history = result.get("history", [])
    best_step = result.get("step")
    soft_hard_gap = next(
        (r["train_loss"] for r in history if r["step"] == best_step), None
    )
    total_edges = int(a_mat.sum() // 2)
    fom_scalars = {
        "success": verification["success"],
        "best_projected_loss": result["value"],
        "best_step": best_step,
        "soft_hard_gap": None if soft_hard_gap is None else round(soft_hard_gap, 4),
        "edge_mismatch": verification["edge_mismatch"],
        "total_edges": total_edges,
        "edge_mismatch_normalized": (
            round(verification["edge_mismatch"] / total_edges, 4) if total_edges else None
        ),
        "projection_method": result.get("projection"),
        "elapsed_seconds": round(result["elapsed_seconds"], 2),
    }
    with (outdir / "fom.json").open("w", encoding="utf-8") as f:
        json.dump(fom_scalars, f, indent=2)

    from .visualize import plot_fom_panel

    plot_fom_panel(history, fom_scalars, outdir / "fom.png")

    print(f"[runs] outputs saved to: {outdir}")
    print(f"[runs] success={verification['success']} "
          f"edge_mismatch={verification['edge_mismatch']} "
          f"elapsed={result['elapsed_seconds']:.1f}s")

    return {
        "cfg": cfg,
        "result": result,
        "verification": verification,
        "a_mat": a_mat,
        "b_mat": b_mat,
        "p_star": p_star,
        "outdir": outdir,
    }
