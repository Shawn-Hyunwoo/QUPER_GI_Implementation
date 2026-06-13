"""CLI entry point for QuPer-GI experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .configs import GIConfig, summarize_config, validate_config
from .graph_data import make_isomorphic_pair, verify_isomorphic_pair
from .metrics import frobenius_mismatch, hidden_perm_overlap, is_success, permutation_validity
from .optimizer import train_quper, train_single_m

_DEFAULTS = GIConfig()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuPer-GI heuristic simulator.")

    parser.add_argument("--num_vertices", type=int, default=_DEFAULTS.num_vertices)
    parser.add_argument("--ancillas", type=int, default=_DEFAULTS.ancillas)
    parser.add_argument("--ansatz", choices=["borel", "bruhat"], default=_DEFAULTS.ansatz)

    parser.add_argument("--device", type=str, default=_DEFAULTS.device)
    parser.add_argument("--diff_method", type=str, default=_DEFAULTS.diff_method)

    parser.add_argument("--graph_p", type=float, default=_DEFAULTS.graph_p)
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument(
        "--perm_source",
        choices=["uniform", "borel", "bruhat"],
        default=_DEFAULTS.perm_source,
        help="Hidden-permutation distribution: uniform over S_n, or sampled "
        "from the m=0 circuit span (guaranteed reachable).",
    )

    parser.add_argument("--steps", type=int, default=_DEFAULTS.steps)
    parser.add_argument(
        "--optimizer",
        choices=["adam", "spsa", "cobyla"],
        default=_DEFAULTS.optimizer,
    )
    parser.add_argument("--lr", type=float, default=_DEFAULTS.lr)
    parser.add_argument("--spsa_a", type=float, default=_DEFAULTS.spsa_a)
    parser.add_argument("--spsa_c", type=float, default=_DEFAULTS.spsa_c)
    parser.add_argument("--spsa_alpha", type=float, default=_DEFAULTS.spsa_alpha)
    parser.add_argument("--spsa_gamma", type=float, default=_DEFAULTS.spsa_gamma)
    parser.add_argument("--cobyla_rhobeg", type=float, default=_DEFAULTS.cobyla_rhobeg)
    parser.add_argument("--cobyla_tol", type=float, default=_DEFAULTS.cobyla_tol)

    parser.add_argument("--lambda_stochastic", type=float, default=_DEFAULTS.lambda_stochastic)
    parser.add_argument("--lambda_entropy", type=float, default=_DEFAULTS.lambda_entropy)
    parser.add_argument("--lambda_orthogonal", type=float, default=_DEFAULTS.lambda_orthogonal)

    parser.add_argument(
        "--projection_interval", type=int, default=_DEFAULTS.projection_interval
    )
    parser.add_argument(
        "--num_random_projections", type=int, default=_DEFAULTS.num_random_projections
    )

    parser.add_argument(
        "--ancilla_schedule",
        action="store_true",
        help="Run the full Algorithm 1 schedule m = 0..ancillas with warm starts, "
        "instead of training only at m = ancillas.",
    )

    parser.add_argument("--outdir", type=str, default="runs/results/quper_gi")

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> GIConfig:
    return GIConfig(
        num_vertices=args.num_vertices,
        ancillas=args.ancillas,
        ansatz=args.ansatz,
        device=args.device,
        diff_method=args.diff_method,
        graph_p=args.graph_p,
        seed=args.seed,
        perm_source=args.perm_source,
        steps=args.steps,
        optimizer=args.optimizer,
        lr=args.lr,
        spsa_a=args.spsa_a,
        spsa_c=args.spsa_c,
        spsa_alpha=args.spsa_alpha,
        spsa_gamma=args.spsa_gamma,
        cobyla_rhobeg=args.cobyla_rhobeg,
        cobyla_tol=args.cobyla_tol,
        lambda_stochastic=args.lambda_stochastic,
        lambda_entropy=args.lambda_entropy,
        lambda_orthogonal=args.lambda_orthogonal,
        projection_interval=args.projection_interval,
        num_random_projections=args.num_random_projections,
    )


def save_outputs(
    outdir: Path,
    cfg: GIConfig,
    result: dict,
    instance: dict[str, np.ndarray] | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    serializable_result = {
        "best_value": result.get("value"),
        "best_step": result.get("step"),
        "best_projection": result.get("projection"),
        "best_ancillas": result.get("ancillas"),
        "final_mismatch": result.get("final_mismatch"),
        "success": result.get("success"),
        "valid_permutation": result.get("valid_permutation"),
        "hidden_overlap": result.get("hidden_overlap"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "stages": result.get("stages"),
        "optimizer_result": result.get("optimizer_result"),
        "history": result.get("history", []),
    }

    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    with (outdir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_result, f, indent=2)

    if result.get("p") is not None:
        np.save(outdir / "best_permutation.npy", result["p"])

    if result.get("theta") is not None:
        np.save(outdir / "best_theta.npy", result["theta"])

    if instance is not None:
        for name, array in instance.items():
            np.save(outdir / f"{name}.npy", array)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = config_from_args(args)

    q, total_wires = validate_config(cfg)

    print("Config:", summarize_config(cfg))
    print(f"Expected total wires: {total_wires}")

    a_mat, b_mat, p_star, perm = make_isomorphic_pair(
        num_vertices=cfg.num_vertices,
        edge_prob=cfg.graph_p,
        seed=cfg.seed,
        perm_source=cfg.perm_source,
    )

    verify_isomorphic_pair(a_mat, b_mat, p_star)

    t_start = time.perf_counter()

    if args.ancilla_schedule:
        result = train_quper(a_mat=a_mat, b_mat=b_mat, p_star=p_star, cfg=cfg)
    else:
        result = train_single_m(a_mat=a_mat, b_mat=b_mat, p_star=p_star, cfg=cfg)

    result["elapsed_seconds"] = time.perf_counter() - t_start

    p_best = result.get("p")

    if p_best is None:
        print("No projected permutation was produced.")
        return

    result["success"] = is_success(a_mat, b_mat, p_best)
    result["valid_permutation"] = permutation_validity(p_best)
    result["hidden_overlap"] = hidden_perm_overlap(p_best, p_star)

    print("Final projected loss:", frobenius_mismatch(a_mat, b_mat, p_best))
    print("Success:", result["success"])
    print("Hidden permutation overlap:", result["hidden_overlap"])
    print("Note: hidden overlap is diagnostic only; automorphisms may allow other valid solutions.")

    outdir = Path(args.outdir)
    instance = {"adjacency_a": a_mat, "adjacency_b": b_mat, "p_star": p_star}
    save_outputs(outdir, cfg, result, instance=instance)
    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
