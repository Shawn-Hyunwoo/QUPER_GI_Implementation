"""Experiment: multi-start success rate at N=16 and N=32.

Reproduces the multi-start probe (same planted bruhat-span instance, only the
initialization seed varies across starts) and saves results under
runs/results/ so they can be revisited.

For each N, runs `--starts` independent Adam runs (default 3) on ONE fixed
instance, recording per-start success (projected GI loss == 0) and the overall
success rate.

    conda run -n quantum_env python -m runs.exp_multistart [--starts 3] [--steps 300]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from quper_gi.configs import GIConfig
from quper_gi.dsm_circuit import QuPerDSM
from quper_gi.graph_data import make_isomorphic_pair, verify_isomorphic_pair
from quper_gi.metrics import (
    edge_mismatch,
    hidden_perm_overlap,
    is_success,
    permutation_validity,
)
from quper_gi.optimizer import initialize_theta, train_single_m


def run_one_size(n: int, n_starts: int, steps: int) -> dict:
    cfg = GIConfig(
        num_vertices=n,
        ancillas=1,
        ansatz="bruhat",
        steps=steps,
        seed=0,
        perm_source="bruhat",
    )

    # One fixed instance; only the initialization differs across starts.
    a_mat, b_mat, p_star, _ = make_isomorphic_pair(
        n, cfg.graph_p, cfg.seed, perm_source="bruhat"
    )
    verify_isomorphic_pair(a_mat, b_mat, p_star)

    model = QuPerDSM(n, cfg.ancillas, cfg.ansatz)
    print(f"\n=== N={n} | {model.total_wires} qubits | {model.num_params} params "
          f"| {n_starts} starts ===")

    starts: list[dict] = []
    best_overall = {"value": float("inf"), "p": None, "start": None}
    t0 = time.perf_counter()

    for s in range(n_starts):
        theta0 = initialize_theta(model.num_params, seed=s)  # init seed = s
        result = train_single_m(a_mat, b_mat, p_star, cfg, theta0=theta0)
        p_best = result["p"]

        rec = {
            "start": s,
            "best_projected_loss": result["value"],
            "best_step": result["step"],
            "projection": result["projection"],
            "edge_mismatch": edge_mismatch(a_mat, b_mat, p_best),
            "valid_permutation": bool(permutation_validity(p_best)),
            "hidden_overlap": hidden_perm_overlap(p_best, p_star),
            "success": bool(is_success(a_mat, b_mat, p_best)),
        }
        starts.append(rec)
        print(f"  start {s}: loss={rec['best_projected_loss']:.6g} "
              f"edges_off={rec['edge_mismatch']} success={rec['success']}")

        if result["value"] < best_overall["value"]:
            best_overall = {"value": result["value"], "p": p_best, "start": s}

    elapsed = time.perf_counter() - t0
    n_success = sum(r["success"] for r in starts)

    summary = {
        "num_vertices": n,
        "qubits": model.total_wires,
        "params": model.num_params,
        "ansatz": cfg.ansatz,
        "ancillas": cfg.ancillas,
        "perm_source": cfg.perm_source,
        "steps": cfg.steps,
        "n_starts": n_starts,
        "n_success": n_success,
        "success_rate": f"{n_success}/{n_starts}",
        "best_loss_overall": best_overall["value"],
        "elapsed_seconds": elapsed,
        "starts": starts,
    }

    outdir = Path(f"runs/results/multistart_N{n}")
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    np.save(outdir / "adjacency_a.npy", a_mat)
    np.save(outdir / "adjacency_b.npy", b_mat)
    np.save(outdir / "p_star.npy", p_star)
    if best_overall["p"] is not None:
        np.save(outdir / "best_permutation.npy", best_overall["p"])

    print(f"  -> {n_success}/{n_starts} solved "
          f"(best loss {best_overall['value']:.6g}, {elapsed:.1f}s) "
          f"saved to {outdir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32])
    args = parser.parse_args()

    overall = []
    for n in args.sizes:
        overall.append(run_one_size(n, args.starts, args.steps))

    print("\n=== overall ===")
    for s in overall:
        print(f"N={s['num_vertices']:>3}: {s['success_rate']} solved "
              f"(best loss {s['best_loss_overall']:.6g})")


if __name__ == "__main__":
    main()
