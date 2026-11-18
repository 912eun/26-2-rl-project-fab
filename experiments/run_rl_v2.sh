#!/bin/bash
# 2026-11-13: PPO 안정화판(검증 체크포인트 + lr 선형 감소 + 300 iter, λ=0.95 통일)과 DQN을 같은 조건으로 시드 0~4 × 앞 공정 정보 유무.
cd "$(dirname "$0")/.."
mkdir -p results/rl_ppo/logs results/rl_dqn/logs
jobs=""
for s in 0 1 2 3 4; do for up in up noup; do jobs="$jobs ppo:$up:$s dqn:$up:$s"; done; done
echo $jobs | tr ' ' '\n' | xargs -P 4 -I{} bash -c '
  IFS=: read alg up s <<< "$1"; flag=""; [ "$up" = noup ] && flag="--no_upstream"
  if [ $alg = ppo ]; then
    python3 -W ignore experiments/ppo_implant.py --iters 300 --lam 0.95 --lam_time --lr_decay --suffix _v2 --seed $s $flag > results/rl_ppo/logs/${up}_v2_s${s}.log 2>&1
  else
    python3 -W ignore experiments/dqn_implant.py --steps 600000 --seed $s $flag > results/rl_dqn/logs/${up}_s${s}.log 2>&1
  fi' _ {}
echo ALL DONE
