#!/bin/bash
# EDA 05: implant 단순 개선 규칙 4종 × 시드 3개 = 2년 실행 12개를 최대 8개씩 돌린다.
# - setsid + caffeinate: 세션이 끝나도 살아남고 유휴 잠자기를 막는다(덮개를 닫으면 멈출 수 있음).
# - 이미 끝난 조합(run_info.txt 있음)은 건너뛰므로, 중간에 죽으면 다시 실행하면 이어서 돈다.
# 사용: bash experiments/run_all_implant_rules.sh
# 진행: tail -qn 1 results/implant_setup/logs/rules_*.log
# 정리: python3 notebooks/05_implant_rules.py
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/implant_setup/logs
if [ "$1" = "--one" ]; then
  name=$2 seed=$3; mode=$name; extra=""
  case $name in hold*) mode=hold; extra="--hold_min ${name#hold}";; esac
  [ -f "results/implant_setup/SMT2020_HVLM_fifo_${name}_Demand_730d_s${seed}/run_info.txt" ] && exit 0
  exec python3 -W ignore experiments/run_implant_setup.py --mode "$mode" $extra --days 730 --seed "$seed" \
    > "results/implant_setup/logs/rules_${name}_s${seed}.log" 2>&1
fi
jobs=""
for seed in 1 2 3; do for name in lssu longest hold15 hold30; do jobs+="$name $seed"$'\n'; done; done
printf '%s' "$jobs" | perl -MPOSIX=setsid -e 'setsid(); exec @ARGV' -- \
  nohup caffeinate -i xargs -P 8 -L 1 bash experiments/run_all_implant_rules.sh --one \
  > results/implant_setup/logs/rules_queue.log 2>&1 &
echo "started queue (pid $!)"
