# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

QuPer-GI: a PennyLane-based heuristic simulator for Graph Isomorphism, training a variational circuit that outputs a doubly-stochastic matrix which is then projected to a hard permutation. The algorithm, fixed wire-layout convention (§3), ansatz definitions (§1, §7), and scaling plan (§8) are all in README.md — read the relevant section before touching circuit code.

## Environment

- Always use the `quantum_env` conda environment. Never create a new virtual environment.
- Conda executable: `/home/HW/miniconda3/bin/conda`
- Runnable commands: prefer `/home/HW/miniconda3/bin/conda run -n quantum_env python ...`

## Working style

- Default to Korean unless explicitly asked otherwise.
- No emojis; minimal formatting; concise and direct.
- 결과 서술은 건조하게. "우스울 정도로", "안정적으로 실패", "킬러", "압도적" 같은
  과장·연출·잘난 척하는 수사 금지. 수치와 사실만 쓰고 평가 형용사는 빼라.
- 작업 크기에 과정을 비례시켜라. 시각화 등 cosmetic한 단순 작업은 국소 수정으로
  끝내고, 코어 모듈 변경·전체 테스트·전체 재실행 같은 풀 사이클을 돌리지 마라.
- Stick to PennyLane; do not introduce alternative quantum SDKs unless explicitly asked.
- When debugging: identify the root cause first, name the exact file(s), propose the smallest fix, and say how to validate it.

## Commands

- Smoke run (small end-to-end check):
  `/home/HW/miniconda3/bin/conda run -n quantum_env python -m quper_gi.run_gi --num_vertices 4 --ancillas 0 --ansatz borel --steps 60 --outdir /tmp/quper_smoke4`
- Experiment runners (organized outputs + visualization under `runs/results/`):
  `conda run -n quantum_env python -m runs.run_smoke` (N=16, minutes) and `python -m runs.run_main256` (N=256, 20 qubits, long — confirm before launching).
- Tests: `conda run -n quantum_env pytest` (testpaths configured in pyproject.toml).
- Lint: `ruff check .` (config in pyproject.toml). Run it on files you edit.

## Non-obvious constraints

- Success = zero **projected** GI loss `||A − P_best B P_bestᵀ||²_F`, NOT recovery of the hidden permutation `P_star` — graphs can have automorphisms.
- The wire layout in README §3 (`anc_ref` / `row` / `anc_u` / `col`) is a fixed convention; do not reorder registers.
- Projection (Hungarian / random-order) is outside the differentiable path — keep it out of the autograd graph.
- Gate endpoint semantics: `param_cx(0)=I`, `param_cx(π)=CNOT`; `param_swap(0)=I`, `param_swap(π)=SWAP`.
- Use `default.qubit` or `lightning.qubit` for small N; `lightning.gpu` only for the N=256 target (optional dependency, commented out in requirements.txt).
- This is a heuristic: failure to reach zero loss does not prove non-isomorphism. Benchmark only on known isomorphic pairs (README §9).
