#!/bin/bash
# 공장 최종 검증 E6 (docs/EXPERIMENT_PLAN.md 6-4절). 학습·검증·시험에 쓰지 않은 공장 시드 4~6, 2년.
# 비교: lssu, cov(--cov_fb, RL 비교 기준과 같은 튜닝값), 알고리즘마다 검증 점수(시드 7)로 고른 정책 1개.
# 진행 조건(미리 정함): E3 시험에서 전체 cov 대비 < 0, 어느 장비군도 +5% 초과 악화 없음, hot lot +5% 초과 악화 없음.
#
# 사용: PATTERN=_e3 ALGS="dqn ppo" WAIT_MIN=10 SCOPE=both bash experiments/run_fab_final.sh
#   PATTERN : 실험 태그 (기본 _e3)   ALGS : 넣을 알고리즘 (기본 dqn)   WAIT_MIN : 기다리기 길이(분), E1에서 고른 값 (기본 0)
#   SCOPE   : each = 장비군 하나씩(나머지는 cov, 주 결과) / all = 세 장비군 동시(보조) / both (기본)
# lssu, cov는 이미 결과가 있으면 다시 돌리지 않는다.
cd "$(dirname "$0")/.." || exit 1
PATTERN=${PATTERN:-_e3}; ALGS=${ALGS:-dqn}; WAIT_MIN=${WAIT_MIN:-0}; SCOPE=${SCOPE:-both}
SCOPES=""; [ $SCOPE != all ] && SCOPES="Implant_128 Implant_132 Implant_91"; [ $SCOPE != each ] && SCOPES="$SCOPES ALL"
mkdir -p results/implant_setup/logs
jobs=""
for s in 4 5 6; do
  [ -f "results/implant_setup/SMT2020_HVLM_fifo_lssu_Demand_730d_s$s/run_info.txt" ] || jobs="$jobs lssu:-:-:$s"
  [ -f "results/implant_setup/SMT2020_HVLM_fifo_covfb_Demand_730d_s$s/run_info.txt" ] || jobs="$jobs cov:-:-:$s"
  while read kind pol up; do
    echo "$ALGS" | grep -qw "$kind" || continue
    for f in $SCOPES; do jobs="$jobs rl:$kind:$pol|$up|$f:$s"; done
  done < <(python3 experiments/pick_policy.py --pattern "$PATTERN")
done
[ -z "$jobs" ] && { echo "돌릴 것 없음"; exit 0; }
echo $jobs | tr ' ' '\n' | WAIT_MIN=$WAIT_MIN PATTERN=$PATTERN xargs -P 8 -I{} bash -c '
  IFS=: read mode kind rest s <<< "$1"
  if [ $mode = lssu ]; then args="--mode lssu"; name=lssu
  elif [ $mode = cov ]; then args="--mode cov --cov_fb"; name=covfb
  else IFS="|" read pol up fam <<< "$rest"; args="--mode rl --kind $kind --policy $pol --wait_min $WAIT_MIN"
       [ $kind = ppo ] && args="$args --sample"; [ $up = noup ] && args="$args --no_upstream"; name=rl_${kind}${PATTERN}_${fam}
       [ $fam != ALL ] && args="$args --rl_fams $fam"
  fi
  caffeinate -i python3 -W ignore experiments/run_implant_setup.py $args --days 730 --seed $s > results/implant_setup/logs/final_${name}_s${s}.log 2>&1' _ {}
echo ALL DONE
