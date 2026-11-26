#!/bin/bash
# 사후 탐색: PPO 갱신 크기 키우기 (처음 학습률 1e-3 / epoch 10), 최종 설정 그대로, 시드 0~2, 4개씩
cd "$(dirname "$0")/.."
PPO="python3 -W ignore experiments/ppo_sb3.py --iters 1000 --decay_iters 600 --lr_floor 0.1 --min_iters 600 --patience 15 --min_delta 0.01 --wait_init_rate_h 0.3 --ent 0.003 --ent_end 0 --wait_min 40"
run() { $PPO $1 --suffix _$2 --seed $3 > results/rl_ppo/logs/up_lt_$2_s$3.log 2>&1; }
for s in 0 1 2; do
  for c in "--lr 1e-3:e4lr1e3" "--K 10:e4k10"; do
    while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 20; done
    run "${c%%:*}" "${c##*:}" $s &
  done
done
wait
echo BIGUPD DONE
