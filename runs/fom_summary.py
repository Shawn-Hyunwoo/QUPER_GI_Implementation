"""Aggregate the immediately-computable FOMs over existing runs/results/.

Scans recursively for run directories (verification.json + result.json +
config.json) and multistart summaries (summary.json), computes per-run FOMs
(journal Sec 5.4) and grouped success rates, prints a table, and writes
runs/results/fom_summary.json.

    conda run -n quantum_env python -m runs.fom_summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("runs/results")


def fom_for_run(d: Path) -> dict | None:
    try:
        ver = json.loads((d / "verification.json").read_text())
        res = json.loads((d / "result.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    history = res.get("history", [])
    best_step = res.get("best_step")
    gap = next((r["train_loss"] for r in history if r.get("step") == best_step), None)

    a_path = d / "adjacency_a.npy"
    total_edges = int(np.load(a_path).sum() // 2) if a_path.exists() else None

    em = ver.get("edge_mismatch")
    return {
        "run": str(d.relative_to(ROOT)),
        "N": cfg.get("num_vertices"),
        "graph_p": cfg.get("graph_p"),
        "success": ver.get("success"),
        "best_projected_loss": res.get("best_value"),
        "best_step": best_step,
        "soft_hard_gap": None if gap is None else round(gap, 3),
        "edge_mismatch": em,
        "total_edges": total_edges,
        "edge_mismatch_normalized": (
            round(em / total_edges, 4) if em is not None and total_edges else None
        ),
        "projection_method": res.get("best_projection"),
        "elapsed_seconds": round(res.get("elapsed_seconds", float("nan")), 1),
    }


def main() -> None:
    runs = []
    multistart = []
    for d in sorted(ROOT.rglob("*")):
        if not d.is_dir():
            continue
        if (d / "verification.json").exists():
            row = fom_for_run(d)
            if row:
                runs.append(row)
        elif (d / "summary.json").exists():
            s = json.loads((d / "summary.json").read_text())
            multistart.append(
                {
                    "run": str(d.relative_to(ROOT)),
                    "N": s.get("num_vertices"),
                    "success_rate": s.get("success_rate"),
                    "best_loss_overall": s.get("best_loss_overall"),
                }
            )

    header = (
        f"{'run':<48} {'N':>3} {'p':>5} {'ok':>3} {'loss':>6} {'step':>5} "
        f"{'gap':>8} {'em/E':>7} {'proj':>14} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        print(
            f"{r['run']:<48} {r['N']:>3} {r['graph_p']:>5.2f} "
            f"{str(r['success'])[:1]:>3} {r['best_projected_loss']:>6.0f} "
            f"{r['best_step']:>5} "
            f"{str(r['soft_hard_gap']):>8} "
            f"{str(r['edge_mismatch_normalized']):>7} "
            f"{str(r['projection_method'])[:14]:>14} {r['elapsed_seconds']:>6}"
        )

    # Grouped success rates S(N, density-tag)
    groups: dict[tuple, list] = {}
    for r in runs:
        tag = "deg4" if "deg4" in r["run"] else "dense"
        groups.setdefault((r["N"], tag), []).append(r["success"])
    print("\nsuccess rate by (N, density), single attempt per instance:")
    for (n, tag), vals in sorted(groups.items()):
        print(f"  N={n:<3} {tag:<6}: {sum(vals)}/{len(vals)}")

    if multistart:
        print("\nmulti-start summaries (one instance, varied init):")
        for m in multistart:
            print(f"  {m['run']:<32} N={m['N']:<3} "
                  f"success {m['success_rate']} best {m['best_loss_overall']}")

    out = {"runs": runs, "multistart": multistart}
    (ROOT / "fom_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten: {ROOT / 'fom_summary.json'}")


if __name__ == "__main__":
    main()
