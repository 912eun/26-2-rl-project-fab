#!/bin/bash
# 배치 전략 실험 8개를 백그라운드로 띄운다.
# - 새 세션(setsid)으로 분리: 터미널/Claude 세션이 끝나도 같이 죽지 않는다.
#   (2026-11-04 첫 실행은 nohup만 써서 세션 종료 때 8개가 함께 죽었다.)
# - caffeinate -i: 실행 중 맥 유휴 잠자기를 막는다(덮개를 닫으면 여전히 멈출 수 있음).
# 사용: bash experiments/run_all_batch_strategies.sh
# 진행: tail -n 2 results/batch_strat/logs/*.log   (10 시뮬레이션일마다 day / elapsed / ETA)
# 결과 정리: python3 notebooks/03_batch_strategies.py
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/batch_strat/logs
for combo in "SMT2020_HVLM fifo" "SMT2020_LVHM cr"; do
  set -- $combo
  for s in Max Min RoundRobin Demand; do
    log="results/batch_strat/logs/${1}_${2}_${s}.log"
    perl -MPOSIX=setsid -e 'setsid(); exec @ARGV' -- \
      nohup caffeinate -i python3 -W ignore experiments/run_batch_strategies.py \
      --dataset "$1" --dispatcher "$2" --strategy "$s" --days 730 --seed 1 > "$log" 2>&1 < /dev/null &
    echo "started $1 $2 $s (pid $!) -> $log"
  done
done
