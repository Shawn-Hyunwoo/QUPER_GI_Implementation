# runs/ — experiment runners

All commands are run **from the repository root** in the `quantum_env` conda
environment. Prefix used below:

> (한국어) 모든 명령은 저장소 루트에서 `quantum_env` 환경으로 실행합니다. 아래의
> `$RUN`은 편의용 접두 변수입니다. `--num-vertices`는 **그래프 정점 수 N**이며
> 큐비트 수가 아닙니다 (큐비트는 레지스터당 q = log₂N, 총 2q+2m 와이어).

```bash
RUN="/home/HW/miniconda3/bin/conda run -n quantum_env"
```

## Command reference

```bash
# Tests (17 tests: gate endpoints, DSM properties, Table-1 span counts,
# projections, N=4 end-to-end)  /  (한국어) 테스트 일괄 실행
$RUN pytest

# Smoke experiments with full visualization (N <= 32). N=2/4/8 solve to zero
# loss within seconds; N=16 typically plateaus (cf. paper Fig. 17/18).
$RUN python -m runs.run_smoke --num-vertices 4
$RUN python -m runs.run_smoke --num-vertices 8
$RUN python -m runs.run_smoke                      # N=16 (default)
$RUN python -m runs.run_smoke --ancilla-schedule   # Algorithm 1: m = 0..ancillas

# Main experiment: N=256, q=8, m=2 -> 20 qubits, Algorithm 1 schedule by default.
# LONG RUN — launch deliberately.
$RUN python -m runs.run_main256
$RUN python -m runs.run_main256 --steps 1000 --ansatz bruhat --seed 0

# Bare CLI (no plots; minimal organized output)
$RUN python -m quper_gi.run_gi --num_vertices 8 --ancillas 1 --ansatz bruhat \
    --steps 300 --perm_source bruhat --outdir runs/results/cli_n8
```

### Common options (both runners)

| Option | Meaning | Default |
|---|---|---|
| `--num-vertices` | N (power of two) — smoke runner only | 16 |
| `--ansatz` | `borel` (q_U + C(q_U,2) params) or `bruhat` (q_U + 3·C(q_U,2)) | bruhat |
| `--ancillas` | ancilla pairs m; total wires = 2·log2(N) + 2m | 1 (smoke) / 2 (main) |
| `--ancilla-schedule` | run Algorithm 1: m = 0 → 1 → … with π/8 warm starts | off (smoke) / **on** (main) |
| `--perm-source` | `bruhat`/`borel`: hidden permutation from the m=0 circuit span (reachable, paper Fig. 18); `uniform`: general instance, mostly outside the span | bruhat (smoke) / uniform (main) |
| `--steps` | Adam iterations per ancilla stage | 300 / 1000 |
| `--seed` | instance + initialization seed | 0 |

Training hyperparameters (Adam lr 0.4, regularizer weights 0.1/0.15/0.01,
projection every step with Hungarian + 50 random-order draws) follow the paper
defaults in `quper_gi/configs.py`; override via `quper_gi.run_gi` flags.

## Output directory

Each run writes a self-contained directory under `runs/results/<name>/`
(gitignored):

> (한국어) 런마다 아래 파일들이 한 폴더에 정리됩니다. 사람이 읽을 요약은
> `report.txt`(동형 판정 + 복원된 매핑 σ), 수치 검증은 `verification.json`,
> 학습 곡선·매칭 그림은 PNG 파일들입니다.

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

> (한국어) 성공 기준은 투영 손실 0 (= 동형 증명서). 숨김 순열과 달라도 자기동형
> 때문에 정답일 수 있고, 실패해도 비동형의 증거가 아닙니다.

The full module-by-module pipeline is documented in `docs/PIPELINE.md`.
