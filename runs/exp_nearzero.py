"""Experiment: near-zero initialization, standard continuous Adam.

The paper initializes theta ~ U(pi/2 +- 0.05) (Algorithm 1), centered between
the {0, pi} endpoints. This probe keeps the optimizer fully continuous but
starts near zero (theta ~ U(-0.05, 0.05)) -- every gate begins almost OFF
(identity), so P_hat starts near the identity permutation and the optimizer
must switch gates on from scratch. One run each at N=8 and N=16, same planted
bruhat-span instances as the other experiments, for direct comparison.

    conda run -n quantum_env python -m runs.exp_nearzero
"""

from __future__ import annotations

import numpy as np

from quper_gi.configs import GIConfig
from quper_gi.dsm_circuit import QuPerDSM
from quper_gi.graph_data import make_isomorphic_pair, verify_isomorphic_pair
from quper_gi.metrics import frobenius_mismatch, is_success
from quper_gi.optimizer import train_single_m


def main() -> None:
    for n in (8, 16):
        cfg = GIConfig(
            num_vertices=n,
            ancillas=1,
            ansatz="bruhat",
            steps=300,
            seed=0,
            perm_source="bruhat",
        )

        a_mat, b_mat, p_star, _ = make_isomorphic_pair(
            n, cfg.graph_p, cfg.seed, perm_source="bruhat"
        )
        verify_isomorphic_pair(a_mat, b_mat, p_star)

        model = QuPerDSM(n, cfg.ancillas, cfg.ansatz)
        rng = np.random.default_rng(cfg.seed)
        theta0 = rng.uniform(-0.05, 0.05, size=model.num_params)

        print(f"\n=== N={n}  (near-zero init, {model.num_params} params) ===")
        result = train_single_m(a_mat, b_mat, p_star, cfg, theta0=theta0)

        loss = frobenius_mismatch(a_mat, b_mat, result["p"])
        print(
            f"N={n}: best projected loss = {result['value']:.6g} "
            f"(step {result['step']}, via {result['projection']}) | "
            f"final mismatch = {loss:.6g} | solved = {is_success(a_mat, b_mat, result['p'])}"
        )


if __name__ == "__main__":
    main()
