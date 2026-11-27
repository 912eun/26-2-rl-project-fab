#!/bin/bash
# FreshPool 수정 확인 (2026-11-16): 점수 비교 + PPO·DQN 짧은 실행이 시험까지 끝나는지. 짧은 실행 결과(_tm)는 끝나고 지운다.
source ~/venvs/rlfab/bin/activate
cd "$(dirname "$0")/.." || exit 1
OUT=results/check_freshpool.txt; : > $OUT
python3 -W ignore experiments/check_pool_same.py >> $OUT 2>&1
t0=$(date +%s)
python3 -W ignore experiments/ppo_sb3.py --iters 20 --val_every 10 --decay_iters 600 --lr_floor 0.1 --wait_init_rate_h 0.3 \
  --ent 0.003 --ent_end 0 --wait_min 5 --suffix _tm --seed 0 > results/check_freshpool_ppo.log 2>&1
echo "PPO 짧은 실행: 종료코드 $? $(( $(date +%s) - t0 ))초, Traceback $(grep -c Traceback results/check_freshpool_ppo.log)" >> $OUT
t0=$(date +%s)
python3 -W ignore experiments/dqn_implant.py --steps 40960 --eps_steps 360000 --nstep_h 0.5 --explore_rate_h 0.6 \
  --wait_min 5 --suffix _tm --seed 0 > results/check_freshpool_dqn.log 2>&1
echo "DQN 짧은 실행: 종료코드 $? $(( $(date +%s) - t0 ))초, Traceback $(grep -c Traceback results/check_freshpool_dqn.log)" >> $OUT
rm -f results/rl_ppo/*_tm_s0* results/rl_dqn/*_tm_s0*
echo "확인 끝" >> $OUT
