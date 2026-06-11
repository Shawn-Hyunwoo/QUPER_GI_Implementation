"""Small-N smoke experiment with full visualization (default N = 16, q = 4).

Note: N (`--num-vertices`) is the GRAPH VERTEX count, not qubits. The circuit
uses q = log2(N) qubits per register, 2q + 2m wires total (N=16, m=1 -> 10).

Usage (from the repository root):

    conda run -n quantum_env python -m runs.run_smoke [--num-vertices 16] \
        [--seed 0] [--steps 300] [--ansatz bruhat] [--ancillas 1] [--ancilla-schedule]

Small enough to run on a laptop CPU in minutes; produces the matching
visualization showing exactly how A's vertices map onto B's.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pennylane import numpy as pnp

from quper_gi.configs import GIConfig
from quper_gi.dsm_circuit import QuPerDSM

from .common import run_experiment
from .visualize import plot_heatmaps, plot_matching, plot_soft_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-vertices",
        type=int,
        default=16,
        help="Power of two. N=8 instances are usually solved exactly; "
        "N=16 typically is not (cf. paper Fig. 17/18).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ansatz", choices=["borel", "bruhat"], default="bruhat")
    parser.add_argument("--ancillas", type=int, default=1)
    parser.add_argument("--ancilla-schedule", action="store_true")
    parser.add_argument(
        "--perm-source",
        choices=["uniform", "borel", "bruhat"],
        default="bruhat",
        help="Default 'bruhat': hidden permutation drawn from the circuit span "
        "(reachable; paper Fig. 18). Use 'uniform' for general instances, which "
        "are mostly outside the m=0 span and usually not exactly solvable.",
    )
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    cfg = GIConfig(
        num_vertices=args.num_vertices,
        ancillas=args.ancillas,
        ansatz=args.ansatz,
        steps=args.steps,
        seed=args.seed,
        perm_source=args.perm_source,
    )

    outdir = Path(
        args.outdir
        if args.outdir
        else f"runs/results/smoke{cfg.num_vertices}_{cfg.ansatz}_m{cfg.ancillas}_"
        f"{cfg.perm_source}_seed{cfg.seed}"
    )

    bundle = run_experiment(cfg, outdir, use_ancilla_schedule=args.ancilla_schedule)
    result = bundle["result"]

    plot_matching(bundle["a_mat"], bundle["b_mat"], result["p"], outdir / "matching.png")
    plot_heatmaps(bundle["a_mat"], bundle["b_mat"], result["p"], outdir / "heatmaps.png")

    # Re-evaluate the soft DSM at the best parameters for inspection.
    ancillas = result.get("ancillas", cfg.ancillas)
    model = QuPerDSM(
        num_vertices=cfg.num_vertices,
        ancillas=ancillas,
        ansatz=cfg.ansatz,
        device_name=cfg.device,
        diff_method=cfg.diff_method,
    )
    p_soft = np.asarray(model(pnp.array(result["theta"], requires_grad=False)))
    np.save(outdir / "p_soft.npy", p_soft)
    plot_soft_matrix(p_soft, result["p"], outdir / "soft_matrix.png")

    print(f"\nReport:\n{(outdir / 'report.txt').read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
