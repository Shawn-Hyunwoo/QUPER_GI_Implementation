# QuPer-GI 의미론적 검토 보고서

- 일자: 2026-06-11
- 대상: QuPer 알고리즘 (arXiv:2505.05981v3, Quantum 게재 확정 2026-01-26) 및 본 저장소 구현
- 범위: 계산적 정체성(Monte Carlo 논쟁), 고전 대비 우위, SGI 확장성 및 구현,
  국소최소 탈출 전략, FOM 후보, 기타 쟁점
- 근거: 논문 전문(40p) 정독 결과 및 본 저장소의 실측 데이터(`runs/results/`, 2026-06-11),
  SGI 빌딩블록 구현(`quper_gi/`, `tests/test_sgi.py`, 2026-06-11)

---

## 0. 요약 (Executive Summary)

QuPer는 O(log n) 큐빗 인코딩과 계산기저-대각 관측량 설계 덕분에 단순한 Monte Carlo
추정으로 환원된다는 비판을 *부분적으로* 회피하나, 그 방어는 "고전적으로 효율 평가가
불가능한 분포족에 대한 샘플링"이라는 미검증 regime(large m)에 의존한다. 고전 대비
우위는 현재 증거로는 확립되지 않았으며, 논문 자체도 이를 주장하지 않는다. GI는
고전 알고리즘이 압도적인 전장이므로 시연용 벤치마크로 이해해야 하고, 실질적 기여
가능성은 SGI·QAP 방향과 large-m regime에 있다. SGI 확장은 회로·투영·옵티마이저를
손대지 않고 손실·인스턴스·판정 함수만 추가하는 방식으로 빌딩블록을 구현했으며
(§3.4), 학습 루프 통합은 미완으로 남겨두었다. N≥16의 국소최소 정체는 양자부가 아닌
고전 옵티마이저의 한계로, 탈출 전략은 대부분 고전 기법이며 적용 전 원인 진단(§4.6)이
선행되어야 한다. FOM으로는 손실 외에 spanned-permutation 수(표현력 계열, 논문 Ch. 5)와
optimality/heuristic gap·성공률(성능 계열, Ch. 6)이 즉시 사용 가능하다.

---

## 1. 계산적 정체성: "Monte Carlo와 동일한 것 아닌가"에 대한 분석

### 1.1 비판이 성립하는 지점 (인정해야 하는 부분)

1. **판독은 순수 MC 추정이다.** 실기기에서 P̂의 n²개 entry는 측정 확률이며, 정밀도
   ε 확보에 O(n²/ε²) 샷이 필요하다. 추정 절차 자체에 양자적 이점은 없다.
2. **m=0이면 전체가 고전 시뮬레이션 가능하다.** U(θ)가 n×n 행렬이므로, 비용 평가에
   이미 필요한 O(n²) 자원으로 P̂_θ를 고전 계산할 수 있다(논문 §5.2). 이 경우 알고리즘은
   "log²n개 파라미터를 가진 고전 휴리스틱"과 계산적으로 등가이다.
3. **논문의 모든 수치 실험(m ≤ 2, 최대 20큐빗)은 exact 고전 시뮬레이션이다.** 따라서
   보고된 성능 중 어느 것도 양자 하드웨어의 이점을 입증하지 않는다.

### 1.2 방어 논리 (구별되는 지점)

MC는 분포에서 표본을 뽑는 행위일 뿐이며, 알고리즘의 정체성은 (i) 분포족의 구조와
(ii) 그 분포족을 누가 효율적으로 평가할 수 있는가에 있다.

1. **군론적으로 통제된 분포족.** 출력은 P̂_θ = Σᵢ λᵢ² Wᵢ⊙W̄ᵢ (Prop. 7) — Bruhat span에
   속한 순열들의 볼록결합이다. 이 족은 ℓ = (q+m) + 3·C(q+m, 2) ∈ O(log²n)개의
   파라미터로 제어되며, span 크기는 2^O((q+m)²)에 이른다(Theorem 6). 순열 위 분포를
   명시적으로 파라미터화하는 고전 휴리스틱(EDA 계열, 논문 [20] 참조)에는 이러한
   압축률("표현력/파라미터" 비율)이 없다.
2. **대각 관측량 설계.** 관측량이 |ij⟩⟨ij| 사영(계산기저 대각)이므로 표본 평균만으로
   P̂이 복원된다. 논문 App. A가 지적하듯, amplitude-encoding 계열(예: 논문 [11])은
   비대각 관측량 탓에 표본을 *사용하는* 고전 후처리가 O(n³)이 되어 이점이 소멸하는데,
   QuPer는 이 함정을 설계 단계에서 회피했다.
3. **양자 주장이 성립하는 regime.** m을 키우면 θ → P̂_θ 사상의 고전 평가 비용은
   operator-Schmidt 항 수와 함께 증가하지만, 양자 샘플링 비용은 m과 무관하게
   O(n²/ε²)로 고정된다(§5.2). 즉 "고전적으로 효율 평가가 불가능한 분포에 대한 MC"가
   되는 영역이 존재하며, 이것이 'genuinely quantum'이라는 논문 표현의 실체이다.

### 1.3 방어의 한계

- 비고전성 ≠ 유용성: large-m 분포족이 *최적화에 유리하다*는 증명·실험 모두 부재.
  논문도 이를 future work(HPC, large m)로 미룬다.
- 하드웨어 학습 비용: gradient를 parameter-shift로 추정하면 스텝당 O(ℓ) 회로 평가가
  추가되어, 전체 표본 복잡도는 단순 판독보다 훨씬 크다.
- 결론: "MC가 아니다"가 아니라 **"고전이 흉내 낼 수 없는 분포족에 대한 MC일 수
  있다(미검증)"**가 정확한 수위이다.

---

## 2. 고전 대비 우위 평가

### 2.1 논문이 실제로 보인 것

| 구분 | 결과 | 출처 |
|---|---|---|
| QAP, tai256c (n=256) | QuPer(Bruhat, m=2) 49,035,944 vs 고전 휴리스틱(FAQ [52]) 98,685,678 vs exact 44,759,294 — **QuPer 우세** | Fig. 22 |
| QAP, esc128 (n=128) | 고전 72 vs QuPer 114–148 (exact 64) — **고전 우세** | Fig. 22 |
| QAP 종합 | n ∈ {16, 64, 256}에서 우세, n=128 열세; "competitive" 수준 | Fig. 13, §6.2.1 |
| GI | n=4–8 경쟁적, n=16부터 normalized heuristic gap ≈ 0.5 — **고전 우세** | Fig. 17 |
| GI, span 내 인스턴스 | 일반 인스턴스보다 개선되나 n=16 미해결 잔존 | Fig. 18 |

본 저장소 실측(2026-06-11, seed 0, span 인스턴스)도 동일 패턴을 재현한다:
N=2/4/8 성공(투영 손실 0; N=8은 step 83, 4.7초), N=16 실패(손실 48–68 정체,
ancilla 스케줄 및 시드 1–3에서도 동일).

### 2.2 판정

1. **"고전보다 확실히 낫다"는 주장은 현재 성립하지 않는다.** 논문의 명시적 주장
   수위는 (i) QAOA 대비 자원 우위 — GI의 QUBO 정식화는 O(n²) 큐빗을 요구하여
   n=256이면 비현실적(§6.3, [62] 인용부)인 반면 QuPer는 20큐빗; (ii) m>0의 경험적
   span boost(Fig. 10–12); (iii) large-m에서의 이점은 *기대*(conjecture)이다.
2. 병목은 양자부가 아니라 고전 옵티마이저이다 — "mainly due to the difficulty of
   the classical optimizer Adam to solve non-convex problems"(§6.3.1)라고 논문이
   반복 자인한다. 이 진술은 §4의 국소최소 논의로 직결된다.
3. GI는 고전 알고리즘(수천–수백만 정점 처리, §6.3 [56, 58–60])이 평정한 문제로,
   이 전장에서의 열세는 알고리즘의 결함이라기보다 벤치마크 선택의 문제이다.

---

## 3. Subgraph Isomorphism(SGI)으로의 확장성과 구현

### 3.1 근거

논문 §6.3: "we could already tackle the harder sub-graph isomorphism problem with
a similar formulation". QuPer의 회로 계보 자체가 SGI 연구(Mariella & Simonetto [12])에서
출발했으며, 논문의 GI 실험은 [12]와의 비교를 위한 설정이다.

### 3.2 정식화

패턴 그래프 H(n_H 정점)를 더미 고립 정점으로 n = 2^q까지 패딩하고, 손실을 패턴의
정점쌍으로 마스킹한다.

```
L(θ) = Σ_{ij} M_ij · (T_ij − (P̂ B P̂ᵀ)_ij)²
```

- **monomorphism**: M = (패턴 엣지에서만 1). 엣지 보존만 강제, 비엣지는 자유.
- **induced subgraph isomorphism**: M = (패턴 정점쌍 전체에서 1). 비엣지도 보존 강제.
- **GI는 특수경우**: M = 전체 1(대각 제외), T = A 이면 기존 GI 손실과 동일하다.

QAP 환원형 tr(W P̂ Dᵀ P̂ᵀ)(W = 패딩된 패턴 인접, D = −B)으로도 동치 표현되므로,
회로·투영·스케줄 등 나머지 파이프라인을 그대로 재사용할 수 있다.

### 3.3 구현 방침: 폴더 구조 불변, 함수 단위 분리

GI/SGI를 별도 서브패키지로 분리하지 않았다. SGI는 GI의 상위 일반화로서 코드의 대부분
(회로 `dsm_circuit.py`/`ansatz.py`/`gates.py`, 투영 `projection.py`, 학습
`optimizer.py`, 시각화)을 공유하며, 분기 지점은 **인스턴스 생성**과 **손실**
두 곳뿐이다. 따라서 폴더 재성형은 과설계로 판단하고, 기존 모듈 내부에 함수만 추가했다.

### 3.4 구현된 빌딩블록 (2026-06-11)

| 모듈 | 추가 함수 | 역할 |
|---|---|---|
| `graph_data.py` | `subgraph_mask(host, pattern_size, induced)` | 제약 마스크 생성 (mono=엣지만, induced=정점쌍 전체, 대각 제외) |
| | `make_subgraph_instance(...)` | host의 앞 pattern_size 정점을 패턴으로, 숨김 순열로 섞어 planted 정답 P* 보장; `(host, B, P*, perm, mask)` 반환 |
| | `verify_subgraph_instance(...)` | planted P*가 마스크 제약을 만족하는지 검증 |
| `losses.py` | `masked_frobenius_loss(p, target, b, mask)` | 마스크 손실 (GI를 특수경우로 포함) |
| | `total_loss(..., mask=None, target_mat=None)` | mask 없으면 GI(동작 불변), 있으면 SGI |
| `metrics.py` | `masked_mismatch`, `subgraph_violations`, `is_sgi_success` | 마스크 손실값, 위반 엣지 수(무방향), 성공 판정 |

검증: `tests/test_sgi.py`(5개) — GI=마스크 특수경우 일치, mono⊆induced 마스크 포함관계,
planted P*가 mono/induced 둘 다 손실·위반 0, 엉뚱한 순열은 위반 발생. 전체 22개 테스트
통과, ruff 클린.

### 3.5 미완 항목 (의도적 분리)

`configs.py`·`optimizer.py`·`projection.py`·`run_gi.py`는 미수정이다. SGI를 *학습으로*
구동하려면 두 통합 지점이 남는다: (i) `train_single_m`이 `mask`/`target`을 받아
`total_loss`로 전달, (ii) `project_best`가 GI 손실 대신 마스크 손실로 best 순열을 선택.
빌딩블록 검증까지를 이번 범위로 한정했으며, 통합 및 `runs/run_sgi.py` 러너는 SGI 실험
착수 시점의 후속 작업으로 둔다.

### 3.6 핵심 난점: 패딩 degeneracy

더미 고립 정점들끼리의 순열은 모두 동치이므로 해의 자유도가 커지고, 손실 지형에
평평한(degenerate) 방향이 다수 생긴다. 이는 §4의 국소최소 문제를 **악화**시키므로,
SGI는 GI보다 최적화가 쉽지 않고 오히려 어렵다. 완화책: 패딩 최소화(n_B를 2의 거듭제곱에
근접시킴), 또는 더미 정점에 미세한 자기루프 가중을 주어 대칭을 인위적으로 깨기.

### 3.7 전략적 평가

SGI는 NP-complete으로, GI(NP-hard로 알려져 있지 않음, §6.3 [57])와 달리 고전
솔버도 대규모에서 한계가 뚜렷하다. 근사 휴리스틱의 기여 여지가 구조적으로 더 크므로,
**GI보다 SGI가 본 접근의 정당화에 유리한 전장**이라고 판단한다.

### 3.8 GI vs SGI 복잡도 관계 — "SGI가 더 쉬워 보인다"는 직관의 교정

문제 성격 정리. 본 저장소의 GI 설정은 planted 복구(동형이 보장된 쌍에서 증명서 P를
탐색)이며, 원래 GI는 결정 문제(동형 여부 yes/no)다. QuPer는 cost 0 발견 시 yes를
증명하지만 no 방향은 아무것도 증명하지 못한다(one-sided, §9).

"후보 부분집합이 많으니 하나만 걸리면 되는 SGI가 GI보다 쉬울 것"이라는 직관은
worst-case 복잡도와 반대다.

1. **GI ≤ SGI (환원).** 패턴과 호스트의 정점 수·엣지 수가 같으면 부분그래프 동형이
   곧 동형이므로, GI는 SGI의 특수경우다. SGI는 최소한 GI만큼 어렵다.
2. **SGI는 NP-complete.** 패턴을 K_k로 잡으면 k-클리크 문제, 길이 n 사이클로 잡으면
   해밀턴 사이클 문제가 되어 표준 NP-complete 문제들을 특수경우로 포함한다. 반면
   GI는 NP-hard로 알려져 있지 않다(논문 §6.3 [57]; 논문도 SGI를 "the harder
   sub-graph isomorphism problem"으로 지칭).
3. **직관이 틀린 이유 — 제약의 경직성과 가지치기.** GI는 두 그래프 크기가 같아
   차수열·이웃 구조가 전역적으로 정확히 일치해야 하며, 이 경직성이 강한 가지치기
   신호가 된다(VF2가 N=32를 ms 단위로 푸는 이유). SGI는 패턴이 어디에나 숨을 수
   있고 monomorphism에서 비엣지가 제약을 주지 않아 필터가 약하다. 후보가 많아
   쉬워 보이게 하는 그 자유도가 가지치기를 무력화하여 어렵게 만든다.
4. **직관이 유효한 영역.** 패턴 크기 k가 작고 고정이면 O(n^k) 전수로 다항시간이고,
   밀집 랜덤 그래프의 소형 패턴(삼각형 등)은 빈번하여 즉시 발견된다. 어려움은
   패턴 크기가 n과 함께 자랄 때(클리크, 해밀턴) 나타난다.

결론: worst-case에서 SGI > GI. 이 비대칭이 §3.7의 판단(고전이 평정한 GI보다,
고전도 막히는 SGI가 휴리스틱의 전장으로 유리함)의 근거다.

---

## 4. 국소최소(local minimum) 탈출 전략

N≥16에서 관측된 정체는 비볼록 지형 + 유효해가 {0,π} 이산점인데 연속 Adam으로
탐색하는 구조에서 기인한다. 병목이 양자부가 아니므로 탈출 전략은 대부분 고전 기법이다.
전략군을 비용·본질성 순으로 정리하되, **적용 전 원인 진단(§4.6)이 선행되어야 한다**.

### 4.1 재시작·앙상블 계열 (저비용, 효과 확실)

- **Multi-start**: 다중 시드 병렬 학습 후 best 선택. 본 저장소에서 시드 스윕으로 이미
  가능. 비볼록 휴리스틱의 표준 1차 방어선.
- **Basin hopping**: 수렴 시 θ에 큰 perturbation을 가하고 재최적화 반복(scipy 제공).
- 논문의 ancilla schedule(m: 0→1→2)도 작은 공간의 해를 큰 공간의 warm-start로 쓰는
  같은 계열이나, 실측상 N=16에는 불충분.

### 4.2 확률성 주입 계열

- 손실 노이즈 또는 정규화 가중치의 epoch별 섭동(안장점 탈출).
- Simulated annealing: 온도 기반으로 손실 증가 이동을 확률 수용(이산 구조와 궁합).
- Langevin: gradient step에 감쇠 노이즈 추가.

### 4.3 이산 구조 직접 공략 (본질적)

유효 파라미터가 {0,π}라는 점을 활용한다.

- **Endpoint coordinate descent**: 각 θᵢ를 0↔π로 번갈아 뒤집는 이산 local search.
  연속 gradient가 못 보는 방향을 탐색.
- **Adam→이산 하이브리드**: Adam 수렴 후 가까운 {0,π} 격자점으로 반올림하고 그곳에서
  이산 search. 본 저장소 투영(soft→hard)과 동일 철학.
- **Tabu / large-neighborhood search**: 최근 방문 순열을 회피. 순열 최적화 고전 문헌
  (논문 [20] EDA 계열)의 주특기.

### 4.4 지형 변형 계열

- **Regularizer annealing**: entropy 정규화를 초반 약→후반 강으로 스케줄링(초반 넓은
  탐색, 후반 순열 수렴). 본 저장소 `lambda_entropy`에 스케줄만 추가하면 됨.
- **Graduated optimization / homotopy**: 부드러운 손실에서 시작해 실제 손실로 연속 변형.
- **Over-parameterization**: m 증대로 지형이 개선될 수 있다는 저자 가설(§6.2.1). 큐빗
  증가로 시뮬 비용↑이나 검증 가치 있음.

### 4.5 적용 우선순위 (N=16 기준, 비용 대비 효과)

1. **Multi-start + entropy annealing** — 코드 거의 그대로, 가장 먼저.
2. **Endpoint coordinate descent를 Adam 후처리로** — 이산 구조 공략, 본질적.
3. 그 외 기법은 §4.6 진단 결과에 따라 선택.

### 4.6 선결 과제: 원인 진단

탈출 기법 선택에 앞서 정체의 원인을 분리해야 한다. 학습 중 gradient norm을 추적하여,

- **landscape 함정**(gradient는 유의미하나 국소최소에 갇힘) → §4.1–4.4 기법이 유효.
- **barren plateau**(gradient가 0에 수렴) → 옵티마이저 교체로는 해결 불가, 초기화·
  안사츠·관측량 설계를 손대야 함.

본 저장소 기준 저비용 실험(기존 학습 루프에 gradient norm 로깅 추가)으로 분리 가능하다.
진단 없이 기법부터 적용하면 헛수고 위험이 있다.

---

## 5. FOM(Figure of Merit) 후보

### 5.1 표현력 계열 (논문 Ch. 5 기반)

| FOM | 정의 | 출처 / 비고 |
|---|---|---|
| Spanned-permutation 수 | θ ∈ {0,π}^ℓ에서 도달 가능한 서로 다른 순열 개수 | Theorem 6, Table 1 (q=2: 24, q=3: 1,344, q=4: 322,560). 본 저장소 `tests/test_span.py`가 q=2를 전수 재현. 큰 q는 샘플링 추정(논문은 4×10⁸ 표본, Fig. 12) |
| Quantum span boost | span(m>0) / span(m=0) | Fig. 10–12의 핵심 메시지를 단일 수치화 |
| 파라미터당 표현력 | ℓ 대비 span 곡선 | Fig. 10–11의 x축; 같은 ℓ에서 m이 크면 회로가 짧음(§5.3) |
| 볼록결합 항 수 w | P̂의 Schmidt 항 수 (상계 2^{mq}, 2^{3mq}) | Prop. 8; 탐색 폭의 proxy |
| 회로 자원 | 큐빗 2q+2m, 깊이 9q−12, 게이트 5·C(q,2), 파라미터 ℓ | Table 1; QAOA O(n²) 큐빗과의 비교축 |

### 5.2 성능 계열 (논문 Ch. 6 기반)

| FOM | 정의 | 출처 / 비고 |
|---|---|---|
| Relative optimality gap | (f(P̃) − f*) / f* | Fig. 13; 최적값을 아는 인스턴스(QAPlib)용 |
| Normalized heuristic gap | (quantum − classical) / (n²/2) | Fig. 17; 최적값 미상 시 고전 휴리스틱 기준 |
| 성공률 | zero-loss 도달 비율 (시드·인스턴스 앙상블) | GI 특화; 본 저장소 `verification.json`의 `success` 집계로 즉시 산출 가능 |
| 손실 곡선 3종 괴리 | DSM 손실 vs LAP-투영 vs random-투영 | Fig. 14/23; soft 최적화와 hard 해 품질의 불일치 진단 |

### 5.3 추가 제안 (논문 외 — 본 보고서의 제안임을 명시)

- steps-to-solution 분포(성공 케이스 한정), edge-mismatch(부분 점수; 구현 완료)
- 투영 잔차 ‖P̂ − Π(P̂)‖_F 또는 P̂ 엔트로피(수렴 sharpness; 엔트로피 정규화 항 재사용)
- 학습 중 방문한 distinct 순열 수(탐색 다양성)
- 양자 비용 축: 회로 평가 횟수 × 샷 수(표본 복잡도) — 고전과의 공정 비교 단위
- SGI 전용: 패턴 엣지 보존율(부분 점수), 패딩 degeneracy로 인한 동치해 수

### 5.4 FOM 구체화 — 계산 방법과 판독 기준 (2026-06-11 추가)

§5.1–5.3을 "무엇을, 어떤 데이터에서, 어떻게 계산하고, 무엇을 결론낼 수 있는가"로
구체화한다.

**현 출력물(`runs/results/*/verification.json`, `result.json`)에서 즉시 계산 가능:**

| FOM | 계산 | 판독 |
|---|---|---|
| 성공률 곡선 S(N, #starts) | `success`를 (N, start 수)별 집계 | 실측: S(16,1)≈0/5, S(16,3)=2/3, S(32,3)=0/3. "multi-start 몇 개로 N이 풀리는가" = 실질 비용 지표 |
| steps-to-solution | 성공 케이스의 `best_step` 분포 | 실측: N=4→0, N=8→83. 회로 평가 횟수(step×(grad+51투영))로 환산하면 고전과 동일 단위 비교 가능 |
| soft–hard gap | 성공 시점의 train_loss − 0 | 실측: smoke8에서 9.4. gap이 크면 투영이 soft 수렴보다 앞서 정답을 찾는다는 뜻 — 안사츠 개선보다 투영 강화가 유리하다는 신호 (논문 Fig. 14의 곡선 분리와 동일 현상) |
| 정규화 edge mismatch | mismatch / 총 엣지 수 | 밀도가 다른 인스턴스(dense vs deg4) 간 공정 비교용 |

**소규모 구현 추가 시 계산 가능:**

| FOM | 계산 | 판독 |
|---|---|---|
| gradient norm 궤적 | `exp_escape.py` 방식의 스텝별 ‖∇ℓ‖ 기록 | norm 유지+loss 정체 = landscape trap, norm 붕괴 = barren plateau. m·N 스케일링을 찍으면 trainability 한계 정량화 |
| P̂ 엔트로피 / 투영잔차 궤적 | 매 스텝 −Σp log p, ‖P̂−Π(P̂)‖_F | soft 해의 sharpness 시계열. soft–hard gap의 동역학 버전 |
| 방문 distinct 순열 수 | 매 스텝 투영 결과 해시 카운트 | random-order 50회가 실제로 다양한 후보를 내는지(App. C 설계 실효성) |

**신규 제안 (기존 목록에 없던 것):**

| FOM | 계산 | 판독 |
|---|---|---|
| span-거리 | 인스턴스의 P*가 m=0 span 내부인지, 외부면 span 내 최근접 순열과의 edge 거리 | 도달 가능성을 인스턴스 단위로 수치화. uniform vs bruhat perm_source 차이를 한 숫자로 |
| 투영 기여도 분해 | best의 `projection` 필드(hungarian vs random_order_k) 비율 집계 | 실측: smoke8은 hungarian. N 증가 시 random-order 기여 변화가 App. C의 실효성 검증 |
| cost-to-solution | (1회 시도 wall-clock) × (1/성공률) | multi-start 포함 기대 해결 시간. 고전 솔버(VF2 ms 단위)와 직접 비교 가능한 유일한 공정 지표 |

---

## 6. 기타 쟁점

1. **벤치마크 공정성**: ER(p=0.5)은 고전 GI에 가장 쉬운 인스턴스 부류이다. 자기동형이
   풍부한 정칙 그래프 등 어려운 부류로 바꾸면 고전·양자 모두 성능이 하락할 것이며,
   비교의 서사가 달라질 수 있다.
2. **노이즈 민감성**: param-cx의 endpoint 정확성(App. B)은 해석적 성질로, 실기기에서는
   P̂이 정확한 DSM이 아니다. Hungarian 투영의 entry-섭동 강건성이 어느 정도 흡수할
   것으로 기대되나 정량 평가는 미수행.
3. **본 저장소의 미구현 항목**(참고): Nesterov-Adam, SEL 안사츠, QAP objective,
   2의 거듭제곱이 아닌 N, SGI 학습 루프 통합(§3.5) — 상세는 `docs/PIPELINE.md`의
   "Known deviations" 표 참조.

---

## 7. 결론

QuPer의 방어 가능한 최대 주장은 다음과 같이 요약된다.

> O(log n) 큐빗 인코딩과 대각 관측량 설계로 "양자 샘플 → 고전 후처리 비용 폭증"이라는
> 전형적 함정은 회피했다. 그러나 비고전성이 최적화 성능으로 이어진다는 증거는 현재
> 없으며, 소규모 실험에서의 성능은 고전 휴리스틱과 비등하거나 열세이다. GI는 시연용
> 전장이며, 실질적 기여 가능성은 (i) SGI·QAP 등 고전 솔버가 약한 문제, (ii) 고전
> 시뮬레이션이 불가능한 large-m regime의 검증에 있다.

후속 작업의 우선순위는 (a) 국소최소 정체의 원인 진단(§4.6) → (b) multi-start +
entropy annealing 및 endpoint 이산 탐색(§4.5) → (c) SGI 학습 루프 통합(§3.5)
순으로 정리된다.
