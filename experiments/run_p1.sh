#!/bin/bash
# E1 P1 담당분(다음 사건까지 + 5분) 실행: DQN·PPO 시드 0~4, 하나씩(WSL 메모리 7.7GB라 동시 실행 불가)
# 실행: 저장소 루트에서 bash experiments/run_p1.sh (WSL, 가상환경 ~/venvs/rlfab)
# 실행 창을 닫으면 WSL이 꺼져 실험도 멈춘다.
#
# 학습·평가 명령은 experiments/run_e1.sh와 글자 그대로 같다. 다른 점 두 가지만 있다.
#  - 하나씩 차례로 돈다(run_e1.sh의 PAR=1과 같음).
#  - 결과 파일(result_*.json)이 이미 있는 실행은 건너뛴다. 중간에 멈춰도 다시 실행하면 이어서 돈다.
#    (2026-11-15: PPO가 메모리 부족으로 멈춰 DQN 다음 사건까지 5개만 끝난 상태에서 재시작하려고 추가)
source ~/venvs/rlfab/bin/activate
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/rl_dqn/logs results/rl_ppo/logs
LOG=results/e1_p1_console.log
echo "시작 $(date '+%F %T')  브랜치 $(git branch --show-current 2>/dev/null)" | tee -a $LOG
for w in 0 5; do
  tag=$([ "$w" = 0 ] && echo _e1ev || echo _e1w$w)
  for alg in dqn ppo; do
    for s in 0 1 2 3 4; do
      if [ $alg = dqn ]; then res=results/rl_dqn/result_up${tag}_s$s.json; else res=results/rl_ppo/result_up_lt${tag}_s$s.json; fi
      if [ -f "$res" ]; then echo "건너뜀(이미 있음): $alg $tag s$s" | tee -a $LOG; continue; fi
      if [ $alg = dqn ]; then
        python3 -W ignore experiments/dqn_implant.py --steps 1200000 --eps_steps 360000 --nstep_h 0.5 --explore_rate_h 0.6 \
          --wait_min $w --suffix $tag --seed $s > results/rl_dqn/logs/up${tag}_s$s.log 2>&1
      else
        python3 -W ignore experiments/ppo_sb3.py --iters 1000 --decay_iters 600 --lr_floor 0.1 --min_iters 600 --patience 15 --min_delta 0.01 \
          --wait_init_rate_h 0.3 --ent 0.003 --ent_end 0 --wait_min $w --suffix $tag --seed $s > results/rl_ppo/logs/up_lt${tag}_s$s.log 2>&1
      fi
      echo "끝: $alg $tag s$s $(grep -c Traceback results/rl_*/logs/up*${tag}_s$s.log | tr '\n' ' ')" | tee -a $LOG
    done
  done
done
for w in 0 5; do
  tag=$([ "$w" = 0 ] && echo _e1ev || echo _e1w$w)
  python3 -W ignore experiments/evaluate.py --kind dqn --wait_min $w --models "results/rl_dqn/dqn_up${tag}_s?.pt" 2>&1 | tee -a $LOG
  python3 -W ignore experiments/evaluate.py --kind ppo --stoch --wait_min $w --models "results/rl_ppo/ppo_up_lt${tag}_s?.pt" 2>&1 | tee -a $LOG
done
echo "E1 0 5 끝 $(date '+%F %T')" | tee -a $LOG
