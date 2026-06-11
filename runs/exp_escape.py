"""Experiment: why does N>=16 get stuck, and can a discrete polish escape it?

Two probes on a single planted GI instance:

  1. Diagnosis -- track the gradient norm during Adam. A norm that decays to
     ~0 while the loss is still high points to a barren-plateau-like regime
     (optimizer swap won't help); a norm that stays sizeable while the loss
     plateaus points to a genuine local-minimum / landscape trap (the discrete
     polish below is the right tool).

  2. Endpoint coordinate descent -- round the Adam-final theta to the nearest
     {0, pi} vertex (where P_hat is an exact permutation) and greedily flip one
     parameter at a time, keeping a flip only if the TRUE projected GI loss
     improves. This searches the discrete {0,pi}^ell space the solution
     actually lives in, rather than the continuous relaxation.

This is an exploratory script (not part of the package API). Run from the repo
root in quantum_env:

    conda run -n quantum_env python -m runs.exp_escape [--num-vertices 16] \
        [--ansatz bruhat] [--ancillas 1] [--steps 300] [--seed 0] [--starts 4]
"""

from __future__ import annotations

import argparse

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from quper_gi.configs import GIConfig
from quper_gi.dsm_circuit import QuPerDSM
from quper_gi.graph_data import make_isomorphic_pair, verify_isomorphic_pair
from quper_gi.losses import total_loss
from quper_gi.optimizer import initialize_theta
from quper_gi.projection import project_best, projected_gi_loss_np


def permutation_2swap_descent(
    perm_mat: np.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    max_passes: int = 30,
) -> tuple[np.ndarray, float, int]:
    """2-opt local search on the PERMUTATION (not on theta).

    Operates directly in permutation space, where the solution lives, sidestep-
    ping the fact that Adam's continuous theta need not lie near the {0,pi} grid
    even when its soft P_hat is near-optimal. Swaps two row assignments and
    keeps the swap iff the true projected GI loss drops. No circuit evaluations.
    """
    sigma = np.argmax(np.asarray(perm_mat), axis=1).astype(int)
    n = sigma.size

    def to_matrix(s: np.ndarray) -> np.ndarray:
        m = np.zeros((n, n))
        m[np.arange(n), s] = 1.0
        return m

    best_loss = projected_gi_loss_np(to_matrix(sigma), a_mat, b_mat)
    evals = 1
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                sigma[i], sigma[j] = sigma[j], sigma[i]
                loss = projected_gi_loss_np(to_matrix(sigma), a_mat, b_mat)
                evals += 1
                if loss < best_loss - 1e-12:
                    best_loss = loss
                    improved = True
                else:
                    sigma[i], sigma[j] = sigma[j], sigma[i]
        if not improved:
            break
    return to_matrix(sigma), best_loss, evals


def run_adam_with_gradnorm(
    model: QuPerDSM,
    theta0: pnp.ndarray,
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    cfg: GIConfig,
) -> dict:
    """Adam loop that also records the gradient norm at each step."""
    a_train = pnp.array(a_mat, requires_grad=False)
    b_train = pnp.array(b_mat, requires_grad=False)

    def objective(t):
        return total_loss(t, model, a_train, b_train, cfg)

    grad_fn = qml.grad(objective)
    opt = qml.AdamOptimizer(stepsize=cfg.lr, beta1=cfg.beta1, beta2=cfg.beta2, eps=cfg.eps)
    rng = np.random.default_rng(cfg.seed + 10_000)

    theta = theta0
    grad_norms: list[float] = []
    best_proj = float("inf")
    best_p = None

    for _step in range(cfg.steps):
        g = grad_fn(theta)
        grad_norms.append(float(np.linalg.norm(np.asarray(g))))
        theta, _ = opt.step_and_cost(objective, theta)

        p_hat = np.asarray(model(theta), dtype=np.float64)
        p_proj, val, _ = project_best(p_hat, a_mat, b_mat, rng, cfg.num_random_projections)
        if val < best_proj:
            best_proj = val
            best_p = p_proj

    return {
        "best_p": best_p,
        "grad_norms": grad_norms,
        "adam_best_proj": best_proj,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-vertices", type=int, default=16)
    parser.add_argument("--ansatz", choices=["borel", "bruhat"], default="bruhat")
    parser.add_argument("--ancillas", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--starts", type=int, default=4, help="multi-start count")
    args = parser.parse_args()

    cfg = GIConfig(
        num_vertices=args.num_vertices,
        ancillas=args.ancillas,
        ansatz=args.ansatz,
        steps=args.steps,
        seed=args.seed,
        perm_source="bruhat",
    )

    a_mat, b_mat, p_star, _ = make_isomorphic_pair(
        cfg.num_vertices, cfg.graph_p, cfg.seed, perm_source="bruhat"
    )
    verify_isomorphic_pair(a_mat, b_mat, p_star)
    model = QuPerDSM(cfg.num_vertices, cfg.ancillas, cfg.ansatz)
    print(model.summary())
    print(f"instance: N={cfg.num_vertices}, planted from bruhat span, starts={args.starts}\n")

    overall_adam = float("inf")
    overall_polish = float("inf")
    adam_solved = polish_solved = 0
    polish_helped = polish_hurt = 0
    failed_gn_ratios: list[float] = []

    for s in range(args.starts):
        theta0 = initialize_theta(model.num_params, cfg.seed + s)
        out = run_adam_with_gradnorm(model, theta0, a_mat, b_mat, cfg)

        gn = out["grad_norms"]
        gn_first = np.mean(gn[:10])
        gn_last = np.mean(gn[-10:])
        adam_best = out["adam_best_proj"]

        # Polish in permutation space, starting from Adam's best projected
        # permutation (theta-space rounding does not work; see the journal).
        polished_p, polish_best, evals = permutation_2swap_descent(
            out["best_p"], a_mat, b_mat
        )
        # sanity: polished result is an exact permutation
        assert np.allclose(polished_p.sum(0), 1) and np.allclose(polished_p.sum(1), 1)

        overall_adam = min(overall_adam, adam_best)
        overall_polish = min(overall_polish, polish_best)
        adam_solved += adam_best == 0.0
        polish_solved += polish_best == 0.0
        polish_helped += polish_best < adam_best - 1e-9
        polish_hurt += polish_best > adam_best + 1e-9
        if adam_best > 0.0:
            failed_gn_ratios.append(gn_last / gn_first)

        print(
            f"start {s}: "
            f"grad_norm {gn_first:.3g} -> {gn_last:.3g} | "
            f"Adam best proj loss = {adam_best:.6g} | "
            f"after 2-swap polish = {polish_best:.6g} "
            f"({evals} loss evals)"
        )

    print("\n--- summary over starts ---")
    print(f"Adam-only   : best={overall_adam:.6g}, solved {adam_solved}/{args.starts}")
    print(f"Adam+polish : best={overall_polish:.6g}, solved {polish_solved}/{args.starts} "
          f"(polish helped {polish_helped}, hurt {polish_hurt})")

    # Diagnosis from the FAILED starts (where the trap actually shows).
    if not failed_gn_ratios:
        print("\ndiagnosis: every start solved -> multi-start alone suffices at this size.")
    elif np.mean(failed_gn_ratios) < 0.05:
        print("\ndiagnosis: on failed starts the gradient norm collapsed "
              "-> barren-plateau-like; better init/restart matters more than polish.")
    else:
        print("\ndiagnosis: on failed starts the gradient norm stayed sizeable while "
              "loss plateaued -> landscape trap (not a barren plateau).")


if __name__ == "__main__":
    main()
