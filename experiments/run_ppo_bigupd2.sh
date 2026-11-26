#!/bin/bash
# 사후 탐색 2: 갱신을 더 키우면 계속 좋아지나 (PPO 처음 학습률 3e-3, 1e-3+epoch 10)와
# 공정 비교용 DQN 학습률 한 단계 키우기(1e-4 → 3e-4). 최종 설정 그대로, 시드 0~2, 4개씩
cd "$(dirname "$0")/.."
PPO="python3 -W ignore experiments/ppo_sb3.py --iters 1000 --decay_iters 600 --lr_floor 0.1 --min_iters 600 --patience 15 --min_delta 0.01 --wait_init_rate_h 0.3 --ent 0.003 --ent_end 0 --wait_min 40"
DQN="python3 -W ignore experiments/dqn_implant.py --steps 1200000 --eps_steps 360000 --nstep_h 0.5 --explore_rate_h 0.6 --wait_min 40"
mkdir -p results/rl_ppo/logs results/rl_dqn/logs
runp() { $PPO $1 --suffix _$2 --seed $3 > results/rl_ppo/logs/up_lt_$2_s$3.log 2>&1; }
rund() { $DQN $1 --suffix _$2 --seed $3 > results/rl_dqn/logs/up_$2_s$3.log 2>&1; }
lim() { while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 20; done; }
for s in 0 1 2; do
  lim; runp "--lr 3e-3" e4lr3e3 $s &
  lim; runp "--lr 1e-3 --K 10" e4lr1e3k10 $s &
  lim; rund "--lr 3e-4" e4lr3e4 $s &
done
wait
echo BIGUPD2 DONE
