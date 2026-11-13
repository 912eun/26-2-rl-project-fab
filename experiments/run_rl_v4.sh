#!/bin/bash
# 2026-11-13 v4: v3 + PPO는 검증·시험을 확률 정책 그대로(argmax가 학습한 교체 규칙을 지웠음), DQN은 시간 기준 다단계 목표 0.5시간(1단계 목표의 과대추정 누적으로 기다리기만 고름).
cd "$(dirname "$0")/.."
mkdir -p results/rl_ppo/logs results/rl_dqn/logs
jobs=""
for s in 0 1 2 3 4; do for up in up noup; do jobs="$jobs ppo:$up:$s dqn:$up:$s"; done; done
echo $jobs | tr ' ' '\n' | xargs -P 4 -I{} bash -c '
  IFS=: read alg up s <<< "$1"; flag=""; [ "$up" = noup ] && flag="--no_upstream"
  if [ $alg = ppo ]; then
    python3 -W ignore experiments/ppo_implant.py --iters 300 --lam 0.95 --lam_time --lr_decay --wait_init 0.99 --ent 0.003 --ent_end 0 --eval_sample --suffix _v4 --seed $s $flag > results/rl_ppo/logs/${up}_v4_s${s}.log 2>&1
  else
    python3 -W ignore experiments/dqn_implant.py --steps 600000 --explore_switch 0.02 --nstep_h 0.5 --suffix _v4 --seed $s $flag > results/rl_dqn/logs/${up}_v4_s${s}.log 2>&1
  fi' _ {}
echo ALL DONE
