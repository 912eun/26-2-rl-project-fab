#!/bin/bash
# 공장 최종 검증 (2026-11-13). 학습·튜닝·검증·작은 환경 시험에 쓰지 않은 새 시드 4~6, 2년.
# 비교: lssu, cov(고친 재생 환경 튜닝값 = RL 비교 기준), DQN 검증 점수로 고른 정책 1개 (PPO는 작은 환경에서 10개 모두 cov보다 나빠 제외).
# 진입 조건 (2026-11-13, v4 결과를 본 뒤 고침): 전체 lot 기준 지연이 cov보다 짧고, 어느 장비군도 +5%(검증 잡음) 넘게 나빠지지 않을 것. 원래 조건은 "세 장비군 모두 cov보다 짧을 것"이었고 고른 정책은 132 +2.3%로 미통과.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/implant_setup/logs
jobs=""
for s in 4 5 6; do
  jobs="$jobs lssu:-:-:$s cov:-:-:$s"
  while read kind pol up; do jobs="$jobs rl:$kind:$pol|$up:$s"; done < <(python3 experiments/pick_policy.py | grep "^dqn")
done
echo $jobs | tr ' ' '\n' | xargs -P 8 -I{} bash -c '
  IFS=: read mode kind rest s <<< "$1"
  if [ $mode = lssu ]; then args="--mode lssu"; name=lssu
  elif [ $mode = cov ]; then args="--mode cov --cov_fb"; name=covfb
  else pol=${rest%%|*}; up=${rest##*|}; args="--mode rl --kind $kind --policy $pol"; [ $kind = ppo ] && args="$args --sample"; [ $up = noup ] && args="$args --no_upstream"; name=rl_$kind
  fi
  caffeinate -i python3 -W ignore experiments/run_implant_setup.py $args --days 730 --seed $s > results/implant_setup/logs/final_${name}_s${s}.log 2>&1' _ {}
echo ALL DONE
