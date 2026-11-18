#!/bin/bash
# cov 규칙을 공장 전체에서 검증한다. 2년, 시드 1~3. 끝난 시드는 건너뛴다.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/implant_setup/logs
for seed in 1 2 3; do
  [ -f "results/implant_setup/SMT2020_HVLM_fifo_cov_Demand_730d_s${seed}/run_info.txt" ] && continue
  perl -MPOSIX=setsid -e 'setsid(); exec @ARGV' -- nohup caffeinate -i python3 -W ignore experiments/run_implant_setup.py \
    --mode cov --days 730 --seed "$seed" > "results/implant_setup/logs/cov_s${seed}.log" 2>&1 < /dev/null &
done
