#!/bin/bash
# EDA 06용 도착 기록: lssu, 240일, 시드 1~3. setsid + caffeinate. 끝난 시드는 건너뛴다.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/implant_setup/logs
for seed in 1 2 3; do
  [ -f "results/implant_setup/SMT2020_HVLM_fifo_lssu_Demand_240d_s${seed}/events_v2.csv" ] && continue
  perl -MPOSIX=setsid -e 'setsid(); exec @ARGV' -- nohup caffeinate -i python3 -W ignore experiments/run_implant_setup.py \
    --mode lssu --days 240 --seed "$seed" --log_events > "results/implant_setup/logs/events_v2_s${seed}.log" 2>&1 < /dev/null &
done
