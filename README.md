# 반도체 fab implant 가스 교체 RL (강화학습개론 2026-2)

SMT2020 공장 시뮬레이터(PySCFabSim-revised)에서 **implant 장비군 하나만 떼어 낸 작은 환경**을 만들고,
거기서 "장비가 비었을 때 다른 가스로 교체할지, 기다릴지"를 강화학습으로 배운 뒤, 배운 정책을 공장 전체에 다시 넣어 확인한다.

## 팀 결정 (3차 회의)

- 주제: implant 가스 교체 RL
- 알고리즘: DQN, PPO (직접 구현). SB3 사용 가능 여부는 교수님께 문의 중
- 환경: 공장 데이터(도착 기록) 기준으로 가져오고, 작은 환경과 공장의 오차를 보정한다 → [오차 보정](#오차-보정-calibration)
- 기다리기 행동: 사건마다 다시 묻기보다 **정해진 시간만큼 기다리기**가 나았다 → `--wait_min` 옵션
- 비교 기준(baseline): **압력 규칙(cov)** → [비교 기준](#비교-기준-baseline)

## 빠른 시작

```bash
pip install -r requirements.txt

# 1) 비교 기준 규칙 (작은 환경, 몇 분)
python3 baselines/run_baselines.py

# 2) 작은 환경이 공장을 얼마나 재현하나 (오차 보정 기준표, 몇 분)
python3 calibration/check_fidelity.py

# 3) RL 학습 + 검증 + 시험 (한 번에 10~20분)
python3 experiments/dqn_implant.py --steps 600000 --explore_switch 0.02 --nstep_h 0.5 --no_upstream --seed 0
python3 experiments/ppo_implant.py --iters 300 --lam 0.95 --lam_time --lr_decay --wait_init 0.99 --ent 0.003 --ent_end 0 --eval_sample --no_upstream --seed 0
#    기다리기를 고정 시간으로: 위 명령에 --wait_min 10 추가

# 4) 공장 전체 시뮬레이터가 필요할 때만 (2년 1회에 1.5시간 이상)
./setup_simulator.sh
python3 experiments/run_implant_setup.py --mode cov --cov_fb --days 730 --seed 4
```

1~3은 레포에 들어 있는 도착 기록(`data/records`)만으로 돌아간다. 시뮬레이터는 4에만 필요하다.

## 폴더 구조

```
src/                    작은 환경
  implant_env.py          도착 기록 재생 이산사건 시뮬레이터, 규칙(lssu, hold, cov=thr(cov=True), rollout)
  implant_gym.py          gymnasium 감싸기: 결정 시점, 상태 29개, 행동 마스크, 보상, 기다리기 옵션(wait_min)
  implant_state.py        상태 계산 (작은 환경과 공장이 같은 함수를 쓴다)
baselines/
  run_baselines.py        lssu / hold / cov 를 시험 조건에서 평가
calibration/
  check_fidelity.py       작은 환경 vs 공장 재현 지표 ①② (오차 보정 기준표)
experiments/
  rl_common.py            망, 검증(체크포인트 고르기), 시험, argmax/확률 행동
  dqn_implant.py          직접 구현 Double DQN (+ 시간 기준 다단계 목표 --nstep_h)
  ppo_implant.py          직접 구현 PPO (+ 행동 마스킹, 시간 기준 할인/GAE)
  run_rl_v4.sh            최종 설정으로 PPO·DQN 각 10개(시드 0~4 × 앞 공정 정보 유무)
  pick_policy.py          검증 점수로 공장에 넣을 정책 고르기
  run_implant_setup.py    공장 전체 실행 (--mode lssu/hold/cov/rl, --log_events 로 도착 기록 뽑기)
  run_implant_logs.sh     도착 기록(240일, 시드 1~3) 다시 뽑기
  run_fab_final.sh        공장 최종 검증 (시드 4~6, 2년)
  run_implant_replay.py   규칙 튜닝(tune)·시험(test)·rollout 비교 (cov의 Q, Tmax를 여기서 골랐다)
  check_state.py          상태 분포 비교(재현 지표 ③), 공장 --log_state 결과 필요
data/
  records/HVLM_lssu_240d_s{1,2,3}/   공장 도착 기록 (gzip). events_v2.csv, prev_times.csv
  fab_reference/{규칙}_730d_s{1,2,3}/  공장 2년 실행의 implant 통계 (재현 지표 비교용)
results/rl_{dqn,ppo}/   최종(v4) 학습 모델(.pt)과 결과(result_*.json)
docs/                   설계 문서. implant_rl_design.html이 전체 정리(브라우저로 열기)
```

## 작은 환경

**데이터.** 공장(HVLM, 디스패칭 FIFO, 확산 배치 Demand, implant 기본 규칙 lssu)을 240일 돌리며 implant 3개 장비군에 lot이 도착한 기록을 남겼다(`events_v2.csv`). 작은 환경은 이 도착을 그대로 재생한다.

| 장비군 | 장비 수 | 교체 시간 | 240일 도착 수 (시드 1) |
|---|---|---|---|
| Implant_128 | 10 | 72분 | 109,565 |
| Implant_132 | 8 | 60분 | 75,472 |
| Implant_91 | 6 | 80분 | 54,799 |

**장비군 안의 규칙은 공장 코드를 옮겼다.** 교체 시간, 최소 7 lot 연속 처리, 처리 매수 기반 PM(가스가 풀림), 지수분포 고장. 왕복 12시간 이하 재방문은 "빨리 처리하면 빨리 돌아오는" 되먹임을 넣으려고 이번 처리 시각에 맞춰 도착시킨다.

**결정 시점.** 장비가 비었고, 지금 가스의 lot이 없고, 다른 가스 lot이 있을 때. 지금 가스 lot이 있으면 무조건 처리한다. hot lot이 있거나 장비에 가스가 없으면(PM·고장 뒤) RL이 아니라 cov가 처리한다.

| | 내용 |
|---|---|
| 상태 s | 29개: 장비군 크기·교체 시간, 기다린 시간, PM까지 남은 양, 다른 장비의 가스별 상태, 가스별 대기 수·hot·가장 오래 기다린 시간, 지난 1시간 도착, 앞 공정 lot 수 6개(`--no_upstream`이면 0) |
| 행동 a | {기다리기, 다른 가스 1로 교체, 다른 가스 2로 교체}. 대기 lot이 없는 가스는 마스크 |
| 보상 r | −(대기 lot 수 × 시간 + 교체 시간) ÷ 장비 수. 결정 사이 흐른 시간만큼 할인(γ = 0.9/시간) |
| 기다리기 | 기본: 다음 사건(도착, 다른 장비 끝 등)에 다시 묻기. `wait_min=10`: 10분 동안 다시 묻지 않기(그사이 지금 가스 lot이 오면 바로 처리) |

## 오차 보정 (calibration)

`python3 calibration/check_fidelity.py`가 기준표를 만든다. 같은 규칙을 공장(2년, 시드 1~3)과 작은 환경에 넣고 implant 대기를 비교한다.

현재 값 (2026-11-13):

| 지표 | 128 | 132 | 91 | 기준 |
|---|---|---|---|---|
| ① 대기 절대값, 재생 − 공장 (시드 1/2/3) | −7 / +4 / +5% | +6 / +3 / **+13%** | +2 / −2 / −5% | ±15% |
| ① 교체 시간 비율 차이 | ±0.5%p | ±0.1%p | ±0.2%p | ±2%p |
| ② lssu 대비 대기 변화의 차이 (hold15, hold30, cov) | −1.4 / +1.5 / 0.0%p | +0.9 / +1.5 / −0.3%p | −1.2 / −1.3 / +1.3%p | ±5%p, 순위 동일 |

- ②가 RL 정책을 공장으로 옮길 때 가장 직접적인 근거라 우선한다. 지금은 모두 통과.
- **남은 오차와 손볼 곳 후보**
  - 132 대기가 공장보다 길게 나온다(특히 시드 3 +13%). 재방문 되먹임 연결 기준 `--max_gap_h`(기본 12시간), 다른 implant 장비군을 거쳐 돌아오는 lot은 고정 도착인 점.
  - 고장 모형: `src/implant_env.py`의 `MTTF`, `MTTR`(지수분포), PM 길이 `PM_DUR`.
  - 동시각 결정 순서: 공장은 같은 시각의 장비를 하나씩 처리해서 "g1 가스로 놀고 있는 장비" 비율이 공장 1~2%, 재생 0%.
  - RL 정책은 아직 ② 비교를 안 했다. 공장 최종 검증(`run_fab_final.sh`)이 그 비교다.
- 도착 기록을 새로 뽑으려면: `./setup_simulator.sh` 후 `bash experiments/run_implant_logs.sh` (시드당 약 30~40분). 기록은 `results/implant_setup/...`에 생기고, `src/implant_env.py`의 `record_path()`가 `data/records`(레포)와 이 폴더를 순서대로 찾는다.

## 비교 기준 (baseline)

`python3 baselines/run_baselines.py`

- **lssu**: SMT2020 기본. 지금 가스 lot 먼저, 없으면 가장 오래 기다린 lot의 가스로 바로 교체.
- **hold T**: 교체가 필요해지면 T분 기다린 뒤 교체.
- **cov (압력 규칙, 팀 기준)**: 가스별 압력 = 대기 lot 수 ÷ (그 가스를 맡은 장비 수 + 1). 가장 높은 압력이 Q 이상이면 그 가스로 교체, 아니면 Tmax분까지 기다린 뒤 교체.
  - 튜닝값(기록 시드 1): 128 = (Q 5, 120분), 132 = (3, 240분), 91 = (7, 240분). `src/implant_gym.py`의 `TUNED_COV`.
  - 팀원 실험의 압력 규칙(Q 3, 2시간)은 `--q 3 --tmax 120`.
  - 공장에서는 `run_implant_setup.py --mode cov --cov_fb`(같은 튜닝값). `--cov_fb` 없이 돌리면 고치기 전 튜닝값이다.

시험 결과 (기록 시드 2·3 × 고장 시드 1·2, 240일), lot당 지연 = 대기 + 교체 시간:

| 정책 | 128 | 132 | 91 |
|---|---|---|---|
| lssu | 2.19h | 2.31h | 3.62h |
| cov | 1.08h | 1.33h | 2.05h |
| DQN 10개 평균 (v4) | 1.01h (−6.9%) | 1.30h (−2.0%) | 2.10h (+2.6%) |
| PPO 10개 평균 (v4) | 1.33h (+23%) | 1.69h (+27%) | 2.60h (+27%) |

## 시드 용도 (데이터 나누기)

| 시드 | 용도 |
|---|---|
| 기록 1 | cov 튜닝, RL 학습, RL 검증(고정 14일 구간 9개) |
| 기록 2, 3 | 작은 환경 시험 (아무것도 고르지 않음) |
| 공장 4, 5, 6 | 공장 최종 검증 (2년) |

같은 시드로 공장을 돌리면 투입·고장 난수가 같아서, 2년 실행의 앞 240일은 같은 시드의 240일 기록과 같은 흐름이다. 그래서 시드마다 용도를 하나만 준다.

## 출처

- SMT2020: Kopp, Hassoun, Kalir & Mönch, IEEE TSM 33(4), 2020.
- PySCFabSim-revised: https://github.com/david-dd/PySCFabSim-revised @ `ae3d55e` (원판 PySCFabSim-release, MIT). 레포에 라이선스 표기가 없어 코드를 복사하지 않고 `setup_simulator.sh`로 받는다. 코드는 수정하지 않고, `experiments/run_implant_setup.py`가 프로세스 안에서 함수를 바꿔 끼운다.
