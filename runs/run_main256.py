"""Main experiment: N = 256 vertices (q = 8), m = 2 ancilla pairs, 20 qubits.

Usage (from the repository root):

    conda run -n quantum_env python -m runs.run_main256 [--seed 0] [--steps 1000] \
        [--ansatz bruhat] [--device default.qubit] [--no-ancilla-schedule] \
        [--projection-interval 10]

Notes
-----
- This reproduces the paper's largest setting (Sec. 2.2): 2*8 + 2*2 = 20 qubits.
- By default the full Algorithm 1 ancilla schedule (m = 0 -> 1 -> 2, warm
  starts) is used, as in the paper.
- `default.qubit` with backprop is the safe default. `lightning.gpu` does not
  support adjoint differentiation of `qml.probs`; if you switch devices, check
  the diff_method (parameter-shift costs 2 circuit evaluations per parameter).
- Expect a long run; the matching plot is skipped at this size (heatmaps only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quper_gi.configs import GIConfig

from .common import run_experiment
from .visualize import plot_heatmaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ansatz", choices=["borel", "bruhat"], default="bruhat")
    parser.add_argument("--ancillas", type=int, default=2)
    parser.add_argument("--device", type=str, default="default.qubit")
    parser.add_argument("--diff-method", type=str, default="best")
    parser.add_argument("--projection-interval", type=int, default=10)
    parser.add_argument("--no-ancilla-schedule", action="store_true")
    parser.add_argument(
        "--perm-source",
        choices=["uniform", "borel", "bruhat"],
        default="uniform",
        help="'uniform' (paper GI setting) or circuit-span sampling "
        "(guaranteed reachable; paper Fig. 18).",
    )
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    cfg = GIConfig(
        num_vertices=256,
        ancillas=args.ancillas,
        ansatz=args.ansatz,
        device=args.device,
        diff_method=args.diff_method,
        steps=args.steps,
        seed=args.seed,
        projection_interval=args.projection_interval,
        perm_source=args.perm_source,
    )

    outdir = Path(
        args.outdir
        if args.outdir
        else f"runs/results/main256_{cfg.ansatz}_m{cfg.ancillas}_"
        f"{cfg.perm_source}_seed{cfg.seed}"
    )

    bundle = run_experiment(
        cfg, outdir, use_ancilla_schedule=not args.no_ancilla_schedule
    )
    result = bundle["result"]

    plot_heatmaps(bundle["a_mat"], bundle["b_mat"], result["p"], outdir / "heatmaps.png")

    print(f"\nReport:\n{(outdir / 'report.txt').read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
