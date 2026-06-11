"""SGI experiment: find a planted pattern embedding in a target graph.

The pattern H is the induced subgraph on the first `pattern-size` vertices of
a random host; the host is scrambled into the target B by a hidden permutation,
so a perfect embedding is guaranteed to exist. Training minimizes the masked
loss (monomorphism: pattern edges only; --induced: all pattern pairs).

    conda run -n quantum_env python -m runs.run_sgi [--num-vertices 8] \
        [--pattern-size 4] [--seed 0] [--steps 300] [--induced]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from quper_gi.configs import GIConfig
from quper_gi.graph_data import make_subgraph_instance, verify_subgraph_instance
from quper_gi.metrics import is_sgi_success, masked_mismatch, subgraph_violations
from quper_gi.optimizer import train_single_m

from .visualize import plot_fom_panel, plot_sgi_matching


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-vertices", type=int, default=8)
    parser.add_argument("--pattern-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--ansatz", choices=["borel", "bruhat"], default="bruhat")
    parser.add_argument("--ancillas", type=int, default=1)
    parser.add_argument("--edge-prob", type=float, default=0.5)
    parser.add_argument("--induced", action="store_true")
    parser.add_argument(
        "--perm-source", choices=["uniform", "borel", "bruhat"], default="bruhat"
    )
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    cfg = GIConfig(
        num_vertices=args.num_vertices,
        ancillas=args.ancillas,
        ansatz=args.ansatz,
        steps=args.steps,
        seed=args.seed,
        graph_p=args.edge_prob,
        perm_source=args.perm_source,
    )

    host, b_mat, p_star, _, mask = make_subgraph_instance(
        num_vertices=cfg.num_vertices,
        pattern_size=args.pattern_size,
        edge_prob=cfg.graph_p,
        seed=cfg.seed,
        perm_source=cfg.perm_source,
        induced=args.induced,
    )
    verify_subgraph_instance(host, b_mat, p_star, mask)

    kind = "induced" if args.induced else "mono"
    outdir = Path(
        args.outdir
        if args.outdir
        else f"runs/results/sgi{cfg.num_vertices}_p{args.pattern_size}_{kind}_"
        f"{cfg.ansatz}_m{cfg.ancillas}_seed{cfg.seed}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    n_constraints = int(mask.sum() // 2)
    print(f"SGI: N={cfg.num_vertices}, pattern={args.pattern_size} ({kind}), "
          f"{n_constraints} constraints")

    t0 = time.perf_counter()
    result = train_single_m(host, b_mat, p_star, cfg, mask=mask)
    elapsed = time.perf_counter() - t0

    p_best = result["p"]
    success = bool(is_sgi_success(host, b_mat, p_best, mask))
    violations = subgraph_violations(host, b_mat, p_best, mask)

    summary = {
        "num_vertices": cfg.num_vertices,
        "pattern_size": args.pattern_size,
        "kind": kind,
        "n_constraints": n_constraints,
        "best_masked_loss": result["value"],
        "best_step": result["step"],
        "projection": result["projection"],
        "violations": violations,
        "success": success,
        "elapsed_seconds": elapsed,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "config.json").write_text(
        json.dumps(asdict(cfg) | {"pattern_size": args.pattern_size, "induced": args.induced},
                   indent=2),
        encoding="utf-8",
    )
    (outdir / "result.json").write_text(
        json.dumps({"history": result.get("history", []), **{
            k: result.get(k) for k in
            ("value", "step", "projection", "final_mismatch", "final_masked_mismatch")
        }}, indent=2),
        encoding="utf-8",
    )
    for name, arr in (("host", host), ("adjacency_b", b_mat), ("p_star", p_star),
                      ("mask", mask), ("best_permutation", p_best)):
        np.save(outdir / f"{name}.npy", arr)

    fom_scalars = {
        "success": success,
        "best_masked_loss": result["value"],
        "best_step": result["step"],
        "violations": violations,
        "n_constraints": n_constraints,
        "projection_method": result.get("projection"),
        "elapsed_seconds": round(elapsed, 2),
    }
    plot_fom_panel(result.get("history", []), fom_scalars, outdir / "fom.png")
    plot_sgi_matching(
        host, b_mat, p_best, mask, args.pattern_size, outdir / "matching.png"
    )

    # Consistency check: stored best value equals a fresh masked evaluation.
    fresh = masked_mismatch(host, b_mat, p_best, mask)
    if not np.isclose(fresh, result["value"]):
        print(f"WARNING: masked loss mismatch (stored {result['value']}, fresh {fresh})")

    print(f"masked loss={result['value']:.6g} (step {result['step']}), "
          f"violations={violations}/{n_constraints}, success={success}, "
          f"{elapsed:.1f}s -> {outdir}")


if __name__ == "__main__":
    main()
