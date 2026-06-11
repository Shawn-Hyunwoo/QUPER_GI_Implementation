# runs/ — experiment runners

Run from the repository root with the `quantum_env` conda environment:

```bash
# N = 16 (q = 4, 10 wires with m = 1): minutes on CPU, full visualization
conda run -n quantum_env python -m runs.run_smoke16

# N = 256 (q = 8, m = 2, 20 wires): the paper's largest setting; long run
conda run -n quantum_env python -m runs.run_main256
```

Each run writes a self-contained directory under `runs/results/<name>/`:

| File | Content |
|---|---|
| `report.txt` | Human-readable summary incl. isomorphism verdict and recovered mapping |
| `verification.json` | valid-permutation check, Frobenius/edge mismatch, success, sigma |
| `result.json` | best loss/step/projection, per-stage results, full history |
| `config.json` | experiment configuration |
| `loss_curve.png` | train loss + projected GI loss vs Adam step |
| `matching.png` | (N <= 32) side-by-side graphs, matched vertices share color/position |
| `heatmaps.png` | A, P B P^T, and residual heatmaps (any N) |
| `soft_matrix.png` | soft DSM P_hat next to its hard projection |
| `*.npy` | A, B, P_star, best permutation, best theta, soft DSM |

Success criterion: zero projected GI loss (`A == P B P^T`), not recovery of
the hidden permutation — automorphisms allow other valid certificates.
A failed run proves nothing (heuristic; see README section 9).
