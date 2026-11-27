#!/bin/bash
# 실행: 저장소 루트에서 bash experiments/run_dqn_e1b.sh (WSL, 가상환경 ~/venvs/rlfab). 진행 기록은 results/rl_dqn/logs/stage2_runner.log
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/rl_dqn/logs
# ② γ (E1b): e1bg08 / e1bg095 각 s0~4, 2개씩 5라운드. result json이 있으면 건너뜀(재실행 시 이어서).
exec >> results/rl_dqn/logs/stage2_runner.log 2>&1
source ~/venvs/rlfab/bin/activate
export PYTHONUNBUFFERED=1
DQN="python -W ignore experiments/dqn_implant.py --steps 1200000 --eps_steps 360000 --nstep_h 0.5 --explore_rate_h 0.6 --wait_min 40"
run() {  # $1 = gamma, $2 = tag, $3 = seed
  if [ -f results/rl_dqn/result_up_$2_s$3.json ]; then echo "skip $2 s$3"; return; fi
  $DQN --gamma_h $1 --suffix _$2 --seed $3 > results/rl_dqn/logs/up_$2_s$3.log 2>&1
}
echo "stage2 start $(date)"
for s in 0 1 2 3 4; do
  echo "round s$s start $(date)"
  run 0.8 e1bg08 $s &
  run 0.95 e1bg095 $s &
  wait
done
echo "all done $(date)"
