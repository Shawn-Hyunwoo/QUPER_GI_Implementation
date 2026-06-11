# QuPer-GI 최적화 실험 및 코드 감사 기록

- 일자: 2026-06-11
- 대상: N=16, N=32에서의 학습 실패 원인 분석, 탈출 전략 실험, 고전 baseline 비교,
  최종 구현 감사
- 산출물: `runs/exp_multistart.py`, `runs/exp_nearzero.py`, `runs/exp_escape.py`,
  `runs/exp_audit.py`, `run_smoke.py`의 `--edge-prob` 옵션
- 데이터: `runs/results/` (gitignore). 실험 시점 seed=0 기준.

---

## 1. 실패 원인 진단: expressivity가 아니라 trainability

N≥16 학습 실패가 표현력 부족(정답이 span 밖)인지 최적화 실패(span 안인데 못 찾음)인지
구분했다. 결론은 후자이며, 근거는 다음 세 가지다.

1. **multi-start 성공.** 같은 N=16 인스턴스에 초기화만 바꿔 여러 번 학습하면
   6-start에서 5/6, 3-start에서 2/3이 cost 0에 도달했다. span 밖이면 어떤 초기화로도
   도달 불가하므로, 정답은 span 안에 있다.
2. **span-내 인스턴스.** `perm_source="bruhat"`로 숨김 순열을 m=0 회로 span에서 뽑아
   도달 가능성이 생성 단계에서 보장된다.
3. **gradient norm 진단.** 실패한 start 0의 gradient norm은 3.09 → 0.903으로 유지됐다
   (0으로 붕괴하지 않음). barren plateau가 아니라 국소최소(landscape trap)다.

`runs/exp_escape.py`가 gradient norm을 학습 중 기록하고, 실패 start의 norm 비율로
barren-plateau / landscape-trap을 구분한다.

---

## 2. 탈출 전략 실험

### 2.1 multi-start — 유효 (`runs/exp_multistart.py`)

같은 인스턴스, 초기화 시드만 다르게.

| N | qubits | params | 성공률 |
|---|---|---|---|
| 16 | 10 | 35 | 2/3 (start 0 실패: loss 48, edge 24) |
| 32 | 12 | 51 | 0/3 (loss 152, 224, 388) |

N=16은 multi-start로 일부 해결되나 N=32는 3-start로 미해결. start 수를 늘려야 할 것으로
보이나 본 세션에서 N=32 대량 start는 실행하지 않았다.

### 2.2 near-zero 초기화 — 역효과 (`runs/exp_nearzero.py`)

표준 연속 Adam에 초기화만 θ ~ U(-0.05, 0.05)로 변경 (논문 기본은 π/2 ± 0.05).

| N | π/2 초기화 | near-zero 초기화 |
|---|---|---|
| 8 | 성공 (loss 0) | 실패 (loss 4) |
| 16 | 실패 (loss 48) | 실패 (loss 64) |

near-zero가 N=8조차 실패시켰다. rx/crx의 손실 gradient가 endpoint(0, π) 근처에서
작아지므로(sin(θ) 형태), 거의 모든 게이트가 gradient-약한 영역에서 출발해 학습 신호가
부족하다. π/2(gradient 최대 지점) 초기화의 타당성을 뒷받침한다.

### 2.3 θ-격자 polish — 실패한 접근 (`runs/exp_escape.py`)

Adam의 best θ를 {0,π} 격자로 반올림 후 1비트씩 뒤집는 좌표하강.

결과: best θ에서 시작해도 Adam이 cost 0으로 푼 start를 polish가 68로 악화시켰다
(6-start 중 polish가 도움 0, 악화 4).

원인: Adam의 좋은 soft 해의 θ는 {0,π} 격자에서 멀다. P̂(측정확률)이 정답 순열이어도
개별 θ는 중간값일 수 있어, 반올림하면 정답과 무관한 꼭짓점으로 떨어진다. soft 해의
품질과 θ의 격자 근접성은 독립적이다.

### 2.4 순열 2-swap polish — 미완

θ가 아니라 순열 공간에서 2-opt local search를 하면 격자 문제를 우회한다. 코드는
`runs/exp_escape.py`의 `permutation_2swap_descent`에 있으나, 마지막 실행이 중단되어
N=16 결과는 미확보다.

---

## 3. 그래프 밀도 실험 (`run_smoke.py --edge-prob`)

평균 degree를 4로 낮춰(ER p = 4/(N-1)) dense(p=0.5)와 비교. seed별로 다른 인스턴스를
단일 시도(인스턴스마다 초기화 1개)로 해결.

| | dense 성공 | dense edge 범위 | deg4 성공 | deg4 edge 범위 |
|---|---|---|---|---|
| N=16 | 0/5 | 24–38 | 0/5 | 20–26 |
| N=32 | 1/5 | 50–192 | 0/5 | 64–74 |

생성된 그래프 실제 평균 degree: N=16 3.88, N=32 3.81.

관찰: degree를 낮춰도 단일 시도 성공률은 0으로 동일. deg4는 dense보다 edge mismatch의
seed 간 편차가 작다(dense N=32는 0–192, deg4는 64–74). 단, 이는 단일 시도 비교이며
multi-start 효과는 측정하지 않았다.

주의: 이 실험은 seed가 인스턴스와 초기화를 동시에 결정하므로 §2.1 multi-start(인스턴스
고정, 초기화만 변경)와 측정 대상이 다르다. §3은 "인스턴스별 단일 시도 성공률",
§2.1은 "한 인스턴스의 multi-start 성공률"이다.

---

## 4. 고전 baseline 비교 (networkx VF2)

QuPer가 실패한 동일 인스턴스를 고전 VF2(백트래킹 매칭)로 해결.

| 인스턴스 | QuPer (300스텝) | VF2 |
|---|---|---|
| N=32 dense seed0 | 미해결 (edge 76) | cost 0, 2.96 ms |
| N=32 deg4 seed0 | 미해결 (edge 68) | cost 0, 2.61 ms |
| N=16 dense seed0 | 미해결 (edge 24) | cost 0, 1.20 ms |

ER 랜덤 그래프는 정점 이웃 구조가 대부분 구별되어 백트래킹 가지치기가 빠르게 수렴하는
부류다. 논문 Fig. 17의 정규화 각주("고전 휴리스틱 결과가 종종 0이라 분모로 못 씀")와
일치한다. naive n! 전수(N=16: 16!≈2×10¹³, N=32: 32!≈2.6×10³⁵)는 비현실적이나, VF2 같은
가지치기 방법으로 이 크기는 밀리초 단위로 해결된다. GI에서 N=16/32는 고전적으로 어려운
영역이 아니다.

---

## 5. 최종 코드 감사 (`runs/exp_audit.py`)

논문 대조에서 아직 코드로 검증하지 않은 지점을 실행 검증. 전부 통과.

| 검증 | 결과 |
|---|---|
| q=3 (N=8) 전수 span, Borel | 64 / 64 (Table 1) |
| q=3 전수 span, Bruhat | 1,344 / 1,344 (Table 1) |
| GI=QAP 전개 항등식 (무작위 P 20개) | 일치 (§6.3) |
| m=2 ancilla DSM 성질 (연속 θ) | 행·열합 1, 음수 없음 |
| Prop. 8 (m=1 endpoint) | 24 hard 순열 + 36 볼록혼합, 전부 DSM |

q=2(24/8)에 이어 q=3에서도 span 수가 논문 Table 1과 정확히 일치한다. 이는 게이트 구성,
Borel/Weyl 워드 순서, 회로 배선, Eq.(3) 스케일링이 동시에 옳아야 나오는 결과다.

정적 재검에서도 결함을 찾지 못했다: Bell 쌍 배선, U 작용 와이어, probs reshape 방향,
param_cx 게이트 순서, param_swap control/target, Hungarian 부호, random-order 의미론,
B=P*ᵀAP* 규약의 전 모듈 일관성, warm-start의 final-θ 사용.

남은 항목은 버그가 아니라 문서화된 편차(`docs/PIPELINE.md` "Known deviations": plain Adam,
SEL 미구현, QAP objective 미구현 등)와 미실행 영역(N=256, lightning.gpu)이다.

---

## 6. 결론

- N≥16 학습 실패는 구현 결함이 아니라 휴리스틱의 최적화 한계이며, 논문 Fig. 17/18과
  정량적으로 일치한다.
- 유효한 탈출 전략은 multi-start(N=16). near-zero 초기화와 θ-격자 polish는 역효과.
  순열 2-swap polish는 미검증.
- 그래프 밀도(degree 4 vs 0.5)는 단일 시도 성공률을 바꾸지 않았다.
- N=16/32는 고전적으로 어려운 문제가 아니다(VF2 밀리초 해결). GI는 본 접근의 우위를
  보일 전장이 아니다.
- 구현은 q=2, q=3 span 전수조사를 포함한 감사를 통과했고, 발견된 결함은 없다.
