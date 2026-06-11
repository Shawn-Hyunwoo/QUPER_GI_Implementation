# QuPer-GI pipeline

End-to-end flow of one experiment, mapped to modules. Paper references are to
arXiv:2505.05981v3 (`docs/QuPer_arXiv-2505.05981.pdf`).

> (한국어) 실험 1회의 전체 흐름을 모듈 단위로 정리한 문서입니다. 인스턴스 생성 →
> 양자 회로(DSM) → 고전 손실 → Adam → 순열 투영 → 검증 → 리포트 순서이며, 각
> 단계 옆에 논문의 해당 절/수식 번호를 달았습니다.

```
graph_data.make_isomorphic_pair          [instance generation]
        │  A (ER graph, Bernoulli p=0.5), hidden P*, B = P*ᵀ A P*
        ▼
optimizer.train_quper / train_single_m   [Algorithm 1]
        │  loop over Adam steps (and ancilla stages m = 0..m*)
        │
        │   ┌─────────────────────────────────────────────────────┐
        │   │ dsm_circuit.QuPerDSM        [quantum side, Fig. 8]  │
        │   │   Bell pairs: (anc_ref↔anc_u), (row↔col)            │
        │   │   ansatz.apply_ansatz on u_wires = anc_u + col:     │
        │   │     X layer (RX) → Borel → [Weyl → Borel]           │
        │   │     gates.param_cx = PhaseShift(θ/2)+CRX(θ) (App. B)│
        │   │   measure probs(row+col) → P̂ = N·reshape (Eq. 3)    │
        │   ├─────────────────────────────────────────────────────┤
        │   │ losses.total_loss           [classical cost, §6.2]  │
        │   │   ‖A − P̂ B P̂ᵀ‖²_F + 0.1·st + 0.15·S_ε + 0.01·ort    │
        │   ├─────────────────────────────────────────────────────┤
        │   │ qml.AdamOptimizer(lr=0.4) updates θ                 │
        │   ├─────────────────────────────────────────────────────┤
        │   │ projection.project_best     [every step, App. C]    │
        │   │   Hungarian (max⟨P,P̂⟩) + 50 random-order (vᵢ=2^i)   │
        │   │   evaluate TRUE GI loss, keep global best P         │
        │   └─────────────────────────────────────────────────────┘
        │
        │  ancilla stage m → m+1: ansatz.transfer_theta
        │  (matching gates copied, new gates initialized at π/8)
        ▼
metrics / runs.common.verify_solution    [verification]
        │  valid permutation? ‖A − P B Pᵀ‖²_F, edge mismatch,
        │  success = zero projected loss, hidden-perm overlap (diagnostic)
        ▼
runs.common + runs.visualize             [reporting]
           report.txt, result/verification/config.json, *.npy,
           loss_curve.png, matching.png (N≤32), heatmaps.png, soft_matrix.png
```

## Stage details

1. **Instance** (`quper_gi/graph_data.py`): ER graph A; hidden permutation
   from `perm_source` — `uniform` over S_N, or `borel`/`bruhat` sampled from
   the m=0 circuit span by drawing θ ∈ {0, π}^ℓ (guaranteed reachable, paper
   Fig. 18). `verify_isomorphic_pair` asserts A = P* B P*ᵀ before training.

   > (한국어) ER 그래프 A를 만들고 숨김 순열 P*로 B를 생성합니다.
   > `perm_source="bruhat"`는 m=0 회로가 실제로 도달 가능한 순열만 뽑는 옵션으로
   > (논문 Fig. 18 설정), 균등 랜덤 순열은 n=16만 돼도 대부분 스팬 밖입니다
   > (|Bruhat span| = 322,560 vs 16! ≈ 2×10¹³).

2. **Circuit** (`quper_gi/dsm_circuit.py`, `ansatz.py`, `gates.py`): wires
   `anc_ref | row | anc_u | col` (2q + 2m total). Hadamard+CNOT Bell pairs,
   U(θ) on anc_u+col, probabilities on row+col. Output P̂ is exactly doubly
   stochastic (unistochastic); with m = 0 and θ ∈ {0,π}^ℓ it is exactly a
   permutation matrix.

   > (한국어) Bell 쌍을 깐 뒤 U(θ)를 anc_u+col에만 적용하고 row+col을 측정하면,
   > 측정 확률 ×N이 정확히 이중확률행렬(DSM) P̂이 됩니다 (논문 Eq. 3). m=0이고
   > 모든 파라미터가 {0, π}이면 P̂은 정확한 순열행렬입니다 — 이 성질은
   > `tests/test_span.py`에서 q=2 전수조사(Bruhat=24개=S₄ 전체, Borel=8개,
   > 논문 Table 1)로 검증됩니다.

3. **Optimization** (`quper_gi/optimizer.py`): θ⁰ ~ U(π/2 ± 0.05). Plain Adam
   (paper uses Adam+Nesterov; known deviation). Early stop when the projected
   loss hits zero. `train_quper` adds the m-schedule with warm starts.

   > (한국어) Algorithm 1 그대로 — 초기화 π/2±0.05, m=0→m* 스케줄, 단계 전환 시
   > 학습된 파라미터를 새 레이아웃으로 옮기고 신규 게이트는 π/8로 초기화합니다.

4. **Projection** (`quper_gi/projection.py`): outside the differentiable path.
   Both methods are scored with the *unregularized* projected GI loss and the
   best-so-far permutation is tracked across all steps and stages.

   > (한국어) 투영은 미분 경로 밖입니다. Hungarian(⟨P,P̂⟩ 최대화)과 random-order
   > (vᵢ=2^i, 50회, 논문 App. C)를 모두 평가해 진짜 GI 손실이 최소인 순열을
   > 전 스텝에 걸쳐 추적합니다.

5. **Verification & reports** (`quper_gi/metrics.py`, `runs/common.py`,
   `runs/visualize.py`): success means an isomorphism certificate
   (A = P B Pᵀ exactly). Failure proves nothing — the heuristic missing zero
   does not imply non-isomorphism (README §9).

   > (한국어) 성공 = 투영 손실 0 (동형 증명서 획득). 숨김 순열 P*와 달라도
   > 자기동형(automorphism) 때문에 유효한 답일 수 있어 overlap은 진단용입니다.
   > 실패는 비동형의 증거가 아닙니다 (휴리스틱).

## Known deviations from the paper / 논문과의 의도적 차이

| Item | Paper | This repo | Impact |
|---|---|---|---|
| Optimizer | Adam + Nesterov momentum (Sec. 6.2) | plain `qml.AdamOptimizer` | PennyLane has no NAdam; convergence may differ slightly. (한국어: PennyLane에 NAdam이 없어 일반 Adam 사용 — 수렴 특성이 미세하게 다를 수 있음) |
| `st` regularizer | column sums only (Sec. 6.2) | row **and** column sums | Identically 0 in exact simulation (P̂ is exactly DSM), so no numerical effect. (한국어: 정확 시뮬레이션에서는 어차피 항상 0이라 차이 없음) |
| P̂ orientation | P̂ = W ⊙ W̄ (Prop. 7) | transposed convention (W ⊙ W̄)ᵀ | The reachable set is identical — the gate group is closed under inverse — so search behavior is unchanged. (한국어: 측정 규약상 전치 방향이지만, 게이트 군이 역원에 닫혀 있어 도달 가능한 순열 집합이 동일) |
| SEL ansatz | third baseline (Sec. 5.3) | not implemented | Only needed for reproducing the paper's ansatz comparison. (한국어: 논문의 안사츠 비교 재현시에만 필요) |
| Objective | general QAP `tr(W P Dᵀ Pᵀ)` + GI | GI Frobenius loss only | The paper's n=256 numbers are QAPlib; direct comparison needs a QAP loss + loader. (한국어: 논문의 256 수치는 QAP 벤치마크라, 직접 대조하려면 QAP 손실 구현 필요) |
| Problem size | GI experiments include n = 6, 12 | N must be a power of two | No padding implemented. (한국어: 2의 거듭제곱이 아닌 N은 패딩 미구현으로 불가) |

## Expected behavior (measured 2026-06-11, seed 0, bruhat-span instances)

> (한국어) 실측 스케일링: N≤8은 수 초 내 정확히 풀리고, N=16부터는 논문(Fig. 17/18)과
> 동일하게 휴리스틱이 정체됩니다. N=256은 "정확한 동형 증명"이 아니라 "손실을 낮춘
> 근사 매칭"이 현실적인 기대치입니다.

| N | q | result |
|---|---|---|
| 2, 4 | 1, 2 | solved at step 0 (trivial/near-trivial) |
| 8 | 3 | solved (~step 83, ≈5 s) |
| 16 | 4 | NOT solved (loss ≈ 48–68 plateau; matches paper Fig. 17/18 gap) |
| 256 | 8 | not yet executed; expect approximate matching, not zero loss — the paper's n=256 result is QAP (≈10% gap), GI experiments stop at n=16 |
