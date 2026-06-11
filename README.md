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

---

## 1. Core idea

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
  test_projection.py
  test_small_gi.py  # end-to-end N=4 recovery + parameter-transfer test

runs/
  common.py         # experiment harness: run, verify, organize outputs
  visualize.py      # matching/heatmap/soft-DSM plots
  run_smoke16.py    # N=16 (q=4) smoke experiment with visualization
  run_main256.py    # N=256 (q=8, m=2, 20 qubits) main experiment
```

Run tests with `pytest`. See `runs/README.md` for the experiment runners.
The reference paper PDF lives at `docs/QuPer_arXiv-2505.05981.pdf` (gitignored).

---

## 5. Installation

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
