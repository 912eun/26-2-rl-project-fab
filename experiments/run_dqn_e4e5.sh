#!/bin/bash
# 실행: 저장소 루트에서 bash experiments/run_dqn_e4e5.sh (WSL, 가상환경 ~/venvs/rlfab). 진행 기록은 results/rl_dqn/logs/stage3_runner.log
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/rl_dqn/logs
# ③ E4 나머지·E5 (DQN, 안내서 5-1 ③): 7조건 × s0~2 = 21개, 2개씩 동시(xargs -P 2, 하나 끝나면 다음 것 시작).
# 최종 설정(40분, lr 1e-4 고정, γ 0.9) 위에 조건 옵션만 바꿈. result json이 있으면 건너뜀(재실행 시 이어서).
exec >> results/rl_dqn/logs/stage3_runner.log 2>&1
source ~/venvs/rlfab/bin/activate
export PYTHONUNBUFFERED=1
run() {  # $1 = 태그, $2 = 시드
  local tag=$1 s=$2 opt pre=up rc
  local BASE="python -W ignore experiments/dqn_implant.py --steps 1200000 --eps_steps 360000 --wait_min 40"
  case $tag in
    e4n025) opt="--nstep_h 0.25 --explore_rate_h 0.6" ;;
    e4n1)   opt="--nstep_h 1 --explore_rate_h 0.6" ;;
    e4x03)  opt="--nstep_h 0.5 --explore_rate_h 0.3" ;;
    e4x12)  opt="--nstep_h 0.5 --explore_rate_h 1.2" ;;
    e5)     opt="--nstep_h 0.5 --explore_rate_h 0.6 --no_upstream"; pre=noup ;;
    e5n0)   opt="--nstep_h 0 --explore_rate_h 0.6" ;;
    e5x)    opt="--nstep_h 0.5" ;;   # 탐색 기준점 없음: --explore_rate_h만 뺌
  esac
  if [ -f results/rl_dqn/result_${pre}_${tag}_s$s.json ]; then echo "skip $tag s$s"; return; fi
  echo "start $tag s$s $(date +%T)"
  $BASE $opt --suffix _$tag --seed $s > results/rl_dqn/logs/${pre}_${tag}_s$s.log 2>&1
  rc=$?
  echo "end $tag s$s $(date +%T) exit=$rc"
}
export -f run
echo "stage3 start $(date)"
for s in 0 1 2; do
  for t in e4n025 e4n1 e4x03 e4x12 e5 e5n0 e5x; do echo "$t $s"; done
done | xargs -P 2 -L 1 bash -c 'run "$0" "$1"'
echo "all done $(date)"
