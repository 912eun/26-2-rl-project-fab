#!/bin/bash
# PPO 학습 시드 확대: 상태(앞 공정 포함/제외) × 학습 시드 1~4, λ는 시간 기준. 4개씩 병렬.
cd "$(dirname "$0")/.."
mkdir -p results/rl_ppo/logs
jobs=""
for s in 1 2 3 4; do for up in up noup; do jobs="$jobs $up:$s"; done; done
echo $jobs | tr ' ' '\n' | xargs -P 4 -I{} bash -c '
  up=${1%%:*}; s=${1##*:}
  flag=""; [ "$up" = noup ] && flag="--no_upstream"
  python3 -W ignore experiments/ppo_implant.py --iters 150 --lam_time --seed $s $flag > results/rl_ppo/logs/${up}_lt_s${s}.log 2>&1
' _ {}
echo ALL_DONE
