#!/bin/bash
# 2026-11-13 v3: 탐색 기준점을 "대부분 기다림"으로. PPO 처음 기다리기 99% + 엔트로피 0.003→0, DQN ε 탐색의 교체 확률 2%.
cd "$(dirname "$0")/.."
mkdir -p results/rl_ppo/logs results/rl_dqn/logs
jobs=""
for s in 0 1 2 3 4; do for up in up noup; do jobs="$jobs ppo:$up:$s dqn:$up:$s"; done; done
echo $jobs | tr ' ' '\n' | xargs -P 4 -I{} bash -c '
  IFS=: read alg up s <<< "$1"; flag=""; [ "$up" = noup ] && flag="--no_upstream"
  if [ $alg = ppo ]; then
    python3 -W ignore experiments/ppo_implant.py --iters 300 --lam 0.95 --lam_time --lr_decay --wait_init 0.99 --ent 0.003 --ent_end 0 --suffix _v3 --seed $s $flag > results/rl_ppo/logs/${up}_v3_s${s}.log 2>&1
  else
    python3 -W ignore experiments/dqn_implant.py --steps 600000 --explore_switch 0.02 --suffix _v3 --seed $s $flag > results/rl_dqn/logs/${up}_v3_s${s}.log 2>&1
  fi' _ {}
echo ALL DONE
