#!/bin/bash
# 실행: 저장소 루트에서 bash experiments/run_dqn_e4lr.sh (WSL, 가상환경 ~/venvs/rlfab). 진행 기록은 results/rl_dqn/logs/stage1_runner.log
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/rl_dqn/logs
exec > results/rl_dqn/logs/stage1_runner.log 2>&1
source ~/venvs/rlfab/bin/activate
export PYTHONUNBUFFERED=1
DQN="python -W ignore experiments/dqn_implant.py --steps 1200000 --eps_steps 360000 --nstep_h 0.5 --explore_rate_h 0.6 --wait_min 40"
LR="--lr_decay_steps 720000 --lr_floor 0.1"
echo "round1 start $(date)"
$DQN $LR --suffix _e4lrdec --seed 0 > results/rl_dqn/logs/up_e4lrdec_s0.log 2>&1 &
$DQN $LR --suffix _e4lrdec --seed 1 > results/rl_dqn/logs/up_e4lrdec_s1.log 2>&1 &
wait
echo "round2 start $(date)"
$DQN $LR --suffix _e4lrdec --seed 2 > results/rl_dqn/logs/up_e4lrdec_s2.log 2>&1 &
$DQN --suffix _chkw40 --seed 0 > results/rl_dqn/logs/up_chkw40_s0.log 2>&1 &
wait
echo "all done $(date)"
