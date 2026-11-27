#!/bin/bash
# 실행: 저장소 루트에서 bash experiments/run_dqn_e3.sh (WSL, 가상환경 ~/venvs/rlfab). 진행 기록은 results/rl_dqn/logs/stage4_runner.log
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/rl_dqn/logs
# ④ E3: 최종 설정, 시드 10~19, 2개씩. 결과 json 있는 시드는 건너뜀.
exec >> results/rl_dqn/logs/stage4_runner.log 2>&1
source ~/venvs/rlfab/bin/activate
export PYTHONUNBUFFERED=1
run() {  # $1 = 시드
  local s=$1 rc
  if [ -f results/rl_dqn/result_up_e3_s$s.json ]; then echo "skip e3 s$s"; return; fi
  echo "start e3 s$s $(date +%T)"
  python -W ignore experiments/dqn_implant.py --steps 1200000 --eps_steps 360000 --nstep_h 0.5 --explore_rate_h 0.6 --wait_min 40 --suffix _e3 --seed $s > results/rl_dqn/logs/up_e3_s$s.log 2>&1
  rc=$?
  echo "end e3 s$s $(date +%T) exit=$rc"
}
export -f run
echo "stage4 start $(date)"
seq 10 19 | xargs -P 2 -L 1 bash -c 'run "$0"'
echo "all done $(date)"
