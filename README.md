# QuPer-GI Skeleton

This repository skeleton is for reproducing the core idea of **QuPer** from:

> *Variational quantum algorithms for permutation-based combinatorial problems: Optimal ansatz generation with applications to quadratic assignment problems and beyond*

The target implementation is a **PennyLane-based heuristic simulator for Graph Isomorphism (GI)**.  
The first goal is not to build a complete GI decision oracle. The first goal is to build an **isomorphic-pair recovery benchmark**:

1. Generate a graph `A`.
2. Generate a hidden permutation `P_star`.
3. Generate an isomorphic graph `B = P_star.T @ A @ P_star`.
4. Train a QuPer-style variational circuit to produce a doubly-stochastic matrix `P_hat`.
5. Project `P_hat` to a hard permutation `P_best`.
6. Declare success if `||A - P_best @ B @ P_best.T||_F^2 == 0`.

For `N = 256` vertices, `q = log2(N) = 8`. With `m = 2` ancilla pairs, the total number of qubits is:

```text
total_wires = 2*q + 2*m = 2*8 + 2*2 = 20
```

The intended backend for the large case is `lightning.gpu`, but the small tests should be run first on `default.qubit` or `lightning.qubit`.

> (한국어) 이 저장소는 위 논문의 QuPer 알고리즘을 PennyLane으로 구현한 그래프
> 동형(GI) 휴리스틱 시뮬레이터입니다. 목표는 GI 판정 오라클이 아니라 "숨김 순열로
> 만든 동형 쌍을 복구하는 벤치마크"이며, 성공 기준은 투영된 GI 손실이 0이 되는
> 것입니다. N=256(q=8, m=2)이면 총 20큐빗입니다.

---

## 1. Core idea

> (한국어) QAOA처럼 비용을 해밀토니안으로 인코딩하지 않고, **이중확률행렬(DSM)을
> 출력하는 변분 회로**를 만들어 비용은 고전적으로 계산합니다. 안사츠는 순열 생성
> 게이트(X, 파라미터화 CX/SWAP)로 구성되며, Bruhat 분해(B·W·B)가 회로 구조의
> 근거입니다. 파라미터 수는 Borel = q+C(q,2), Bruhat = q+3·C(q,2).

The original QuPer paper studies permutation-based optimization problems:

```math
\min_{P \in \Pi_N} f(P)
```

where `P` is an `N x N` permutation matrix.

Instead of encoding the cost function as a Hamiltonian, QuPer builds a variational circuit that outputs a doubly-stochastic matrix:

```math
\hat P_\theta \in \mathrm{DSM}(N)
```

The cost is then evaluated classically:

```math
f(\hat P_\theta)
```

and the final soft matrix is projected to a hard permutation:

```math
\hat P_\theta \rightarrow \tilde P \in \Pi_N
```

The ansatz is based on permutation-generating gates, especially `X`, parametrized `CX`, and parametrized `SWAP`. The Bruhat-style ansatz is organized as:

```text
X layer -> Borel block -> Weyl block -> Borel block
```

The Borel-only ansatz is:

```text
X layer -> Borel block
```

For a graph with `N = 2^q` vertices, the QuPer DSM circuit uses:

```text
2*q + 2*m
```

wires, where `m` is the number of ancilla pairs.

---

## 2. Applying QuPer to GI

> (한국어) GI를 연속 완화 손실 ‖A − P̂BP̂ᵀ‖²_F 로 풀고, 학습 중 주기적으로 P̂을
> 진짜 순열로 투영해 평가합니다. 성공 판정은 숨김 순열 일치가 아니라 투영 손실
> 0입니다 — 그래프 자기동형 때문에 다른 순열도 유효한 답일 수 있기 때문입니다.

For two undirected graphs with adjacency matrices:

```math
A, B \in \{0,1\}^{N \times N}
```

GI asks whether there exists a permutation matrix `P` such that:

```math
A = P B P^T
```

Use the continuous relaxation:

```math
L_{\mathrm{GI}}(\theta)
=
\|A - \hat P_\theta B \hat P_\theta^T\|_F^2
```

During training, periodically project `P_hat` to a hard permutation `P_proj` and evaluate:

```math
\|A - P_{\mathrm{proj}} B P_{\mathrm{proj}}^T\|_F^2
```

Success is defined by zero projected GI loss, not by matching the hidden `P_star`, because graphs can have automorphisms.

---

## 3. Wire convention

> (한국어) 와이어 배치는 고정 규약입니다: anc_ref | row | anc_u | col 순서로 총
> 2q+2m개. U(θ)는 anc_u+col에만 작용하고 row+col을 측정합니다. 측정 확률에 ×N을
> 곱해야 행/열 합이 1인 DSM이 됩니다 (논문 Eq. 3) — 아래 수식 참조.

This skeleton fixes the following wire layout.

```text
N = number of vertices
q = log2(N)
m = number of ancilla pairs
total_wires = 2*q + 2*m

anc_ref: [0, ..., m-1]
row:     [m, ..., m+q-1]
anc_u:   [m+q, ..., 2*m+q-1]
col:     [2*m+q, ..., 2*m+2*q-1]

U(theta) acts on:
u_wires = anc_u + col

Measure probabilities on:
matrix_wires = row + col
```

Then:

```text
len(qml.probs(wires=row+col)) = 2^(2q) = N^2
```

and the result is reshaped and rescaled into:

```text
P_hat = N * probs.reshape(N, N)
```

The factor `N` comes from the paper's Eq. (3), `p_ij = N * Tr(rho |ij><ij|)`:
measured probabilities sum to 1 over the `N^2` outcomes, so the rescaling makes
every row and column of `P_hat` sum to 1 (doubly stochastic).

---

## 4. File layout

> (한국어) `quper_gi/`는 알고리즘 패키지, `tests/`는 pytest 테스트(17개),
> `runs/`는 실험 러너·시각화입니다. 명령어 모음은 `runs/README.md`, 모듈별
> 파이프라인과 논문과의 차이점은 `docs/PIPELINE.md` 참조.

```text
quper_gi/
  __init__.py
  configs.py        # config dataclass and validation
  graph_data.py     # random graph and hidden permutation generation
  gates.py          # parametrized-CX and parametrized-SWAP
  ansatz.py         # Borel, Weyl, Bruhat ansatz builders
  dsm_circuit.py    # QuPer DSM-producing circuit
  losses.py         # GI loss and regularizers
  projection.py     # Hungarian and random-order projection
  optimizer.py      # Adam training loop
  metrics.py        # success metrics and logging helpers
  run_gi.py         # CLI entry point

tests/
  test_gates.py     # endpoint tests: param_cx/param_swap at 0 and pi
  test_dsm.py       # DSM shape/stochasticity/permutation-endpoint tests
  test_span.py      # q=2 exhaustive span counts vs paper Table 1 (24 / 8)
  test_projection.py
  test_small_gi.py  # end-to-end N=4 recovery + parameter-transfer test

runs/
  common.py         # experiment harness: run, verify, organize outputs
  visualize.py      # matching/heatmap/soft-DSM plots
  run_smoke.py      # small-N smoke experiment with visualization (--num-vertices, default 16)
  run_main256.py    # N=256 (q=8, m=2, 20 qubits) main experiment
```

Run tests with `pytest`. See `runs/README.md` for the experiment runners and
the full command reference, and `docs/PIPELINE.md` for the module-by-module
pipeline. The reference paper PDF lives at `docs/QuPer_arXiv-2505.05981.pdf`
(gitignored).

---

## 5. Installation

> (한국어) 이 머신에서는 가상환경을 새로 만들지 말고 기존 `quantum_env` conda
> 환경을 사용하세요 (CLAUDE.md 참조):
> `/home/HW/miniconda3/bin/conda run -n quantum_env python ...`

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate

pip install pennylane numpy scipy
# Optional GPU backend, depending on your CUDA/PennyLane setup:
pip install pennylane-lightning-gpu
```

---

## 6. First small run

> (한국어) 기본 하이퍼파라미터는 모두 논문값입니다 — Adam lr 0.4, 정규화 가중치
> 0.1/0.15/0.01, 매 스텝 투영(Hungarian + random-order 50회). Algorithm 1의
> m-스케줄(warm start, 신규 게이트 π/8)은 `--ancilla_schedule`로 켭니다.

Defaults follow the paper: Adam step size 0.4 (Sec. 6.3), regularizer weights
`st=0.1, entropy=0.15, orthogonality=0.01` (Sec. 6.2), projection at every
iteration with Hungarian + 50 random-order draws (Algorithm 1, Appendix C).
Pass `--ancilla_schedule` to run the full Algorithm 1 loop `m = 0..ancillas`
with warm-started parameters (new gates initialized at `pi/8`).

Start with `N=4` or `N=8`.

```bash
python -m quper_gi.run_gi \
  --num_vertices 4 \
  --ancillas 0 \
  --ansatz borel \
  --device default.qubit \
  --diff_method best \
  --steps 50 \
  --projection_interval 5
```

Then scale gradually:

```bash
python -m quper_gi.run_gi \
  --num_vertices 16 \
  --ancillas 1 \
  --ansatz bruhat \
  --device default.qubit \
  --diff_method best \
  --steps 200
```

Large target:

```bash
python -m quper_gi.run_gi \
  --num_vertices 256 \
  --ancillas 2 \
  --ansatz bruhat \
  --device lightning.gpu \
  --diff_method best \
  --steps 1000 \
  --projection_interval 10
```

---

## 7. Implementation notes

> (한국어) 파라미터화 CX는 PhaseShift(θ/2)+CRX(θ) 구성으로 θ=π에서 위상 잔여
> 없이 정확히 CNOT이 됩니다 (논문 App. B; `tests/test_gates.py`에서 행렬 단위
> 검증). SWAP은 CNOT·param_cx(역방향)·CNOT. 투영은 미분 경로 밖에서 Hungarian과
> random-order(vᵢ=2^i) 둘 다 시도해 진짜 손실 최소를 취합니다.

### 7.1 Parametrized-CX

The intended switch behavior is:

```text
param_cx(0)  -> identity
param_cx(pi) -> CNOT
```

The skeleton uses:

```text
PhaseShift(theta/2) on control
CRX(theta) on [control, target]
```

This is designed so that the endpoint at `theta = pi` behaves as CNOT up to the intended phase correction.

### 7.2 Parametrized-SWAP

The skeleton uses:

```text
CNOT(a,b)
param_cx(phi, b, a)
CNOT(a,b)
```

Endpoint behavior:

```text
param_swap(0)  -> identity
param_swap(pi) -> SWAP
```

### 7.3 Loss

The base loss is:

```math
\|A - \hat P B \hat P^T\|_F^2
```

Optional regularizers:

```text
row/column stochasticity
entropy sharpening
orthogonality
```

### 7.4 Projection

Projection is outside the differentiable path.

Default projection:

```text
Hungarian assignment maximizing <P, P_hat>
```

Optional projection:

```text
random-order projection from the paper-inspired idea
```

The best projected permutation is selected by true projected GI loss.

---

## 8. Development plan

> (한국어) 아래 Phase 1–4는 구현 완료 상태입니다 — Phase 1·2는 `tests/`에서 검증
> (게이트 endpoint, DSM 성질, 추가로 q=2 span 전수조사 = 논문 Table 1 재현),
> Phase 3·4는 실측 결과 N=2/4/8 성공(투영 손실 0), N=16 정체(논문 Fig. 17/18과
> 일치). Phase 5(N=256, 20큐빗)는 러너(`runs/run_main256.py`)만 준비되어 있고
> 아직 실행 전입니다. 상세 실측 표는 `docs/PIPELINE.md` 참조.

### Phase 1: Endpoint tests

Verify:

```text
param_cx(0) == I
param_cx(pi) == CNOT
param_swap(0) == I
param_swap(pi) == SWAP
```

### Phase 2: DSM tests

Verify for small `N`:

```text
P_hat.shape == (N,N)
P_hat row sums ~= 1
P_hat col sums ~= 1
P_hat entries >= 0
```

### Phase 3: Tiny GI recovery

Run:

```text
N = 4
m = 0
ansatz = borel
```

Goal:

```text
projected GI loss reaches 0 for easy random instances
```

### Phase 4: Scale

Try:

```text
N=8, 16
m=0,1,2
ansatz=borel, bruhat
```

### Phase 5: 256-node experiment

Try:

```text
N=256
q=8
m=2
total_wires=20
ansatz=borel first, then bruhat
```

Borel parameter count:

```text
qU = q + m = 10
params = qU + C(qU,2) = 55
```

Bruhat parameter count:

```text
params = qU + 3*C(qU,2) = 145
```

---

## 9. Important caution

This is a heuristic recovery algorithm. If the optimizer fails to find zero loss, that does not prove non-isomorphism.

For the first implementation, only benchmark on known isomorphic graph pairs generated by a hidden permutation.

> (한국어) 이 알고리즘은 휴리스틱입니다. 손실이 0에 도달하지 못해도 두 그래프가
> 비동형이라는 증거가 아닙니다. 반드시 숨김 순열로 생성한 "동형이 보장된 쌍"으로만
> 벤치마크하세요. 또한 균등 랜덤 순열은 회로 스팬 밖일 수 있으므로, 도달 가능성을
> 보장하려면 `--perm_source bruhat`을 사용하세요 (논문 Fig. 18 설정).
