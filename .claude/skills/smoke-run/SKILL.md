---
name: smoke-run
description: Run a tiny QuPer-GI experiment (N=4, borel ansatz, 50 steps) end-to-end and check its output artifacts. Use after code changes to quper_gi/ to verify the pipeline still runs, since the unit tests are mostly placeholders.
---

Run a minimal QuPer-GI experiment and verify its outputs.

1. Run (use a throwaway output directory):

```bash
/home/HW/miniconda3/bin/conda run -n quantum_env python -m quper_gi.run_gi \
  --num_vertices 4 \
  --ancillas 0 \
  --ansatz borel \
  --device default.qubit \
  --diff_method best \
  --steps 50 \
  --projection_interval 5 \
  --outdir /tmp/quper_smoke
```

2. Check the run:
   - The command exits without an exception.
   - `/tmp/quper_smoke/` contains `config.json`, `result.json`, `best_permutation.npy`, `best_theta.npy`.
   - Read `result.json` and report the best projected GI loss and the step it occurred at.
   - Load `best_permutation.npy` and confirm it is a valid 4×4 permutation matrix (0/1 entries, one 1 per row and column).

3. Interpret honestly:
   - A crash, NaN loss, wrong matrix shape, or invalid permutation = pipeline broken; report the traceback/values.
   - A non-zero final projected loss is NOT necessarily a failure — this is a heuristic optimizer and N=4 runs can miss. If the pipeline ran cleanly but loss stayed > 0, say exactly that and optionally retry once with a different `--seed` before drawing conclusions.

Report the actual numbers from `result.json`, citing the file path.
