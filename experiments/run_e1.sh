#!/bin/bash
# E1 담당분 실행 (docs/E1_wait_length_design.md E1-8, 명령은 docs/EXPERIMENT_PLAN.md 3-1).
# 사용: bash experiments/run_e1.sh <길이...>      길이: 0(다음 사건까지) 5 10 20 40 80
#   예) P3: bash experiments/run_e1.sh 40 80
# 길이마다 DQN·PPO 시드 0~4를 학습하고(4개씩 동시에), 끝나면 시험 평가(evaluate.py)까지 한다.
cd "$(dirname "$0")/.." || exit 1
mkdir -p results/rl_dqn/logs results/rl_ppo/logs
jobs=""
for w in "$@"; do
  for s in 0 1 2 3 4; do jobs="$jobs dqn:$w:$s"; done
  for s in 0 1 2 3 4; do jobs="$jobs ppo:$w:$s"; done
done
echo $jobs | tr ' ' '\n' | xargs -P ${PAR:-4} -I{} bash -c '
  IFS=: read alg w s <<< "$1"; tag=$([ "$w" = 0 ] && echo _e1ev || echo _e1w$w)
  if [ $alg = dqn ]; then
    python3 -W ignore experiments/dqn_implant.py --steps 1200000 --eps_steps 360000 --nstep_h 0.5 --explore_rate_h 0.6 \
      --wait_min $w --suffix $tag --seed $s > results/rl_dqn/logs/up${tag}_s$s.log 2>&1
  else
    python3 -W ignore experiments/ppo_sb3.py --iters 1000 --decay_iters 600 --lr_floor 0.1 --min_iters 600 --patience 15 --min_delta 0.01 \
      --wait_init_rate_h 0.3 --ent 0.003 --ent_end 0 --wait_min $w --suffix $tag --seed $s > results/rl_ppo/logs/up_lt${tag}_s$s.log 2>&1
  fi
  echo "끝: $alg $tag s$s $(grep -c Traceback results/rl_*/logs/up*${tag}_s$s.log | tr "\n" " ")"' _ {}
for w in "$@"; do
  tag=$([ "$w" = 0 ] && echo _e1ev || echo _e1w$w)
  python3 -W ignore experiments/evaluate.py --kind dqn --wait_min $w --models "results/rl_dqn/dqn_up${tag}_s?.pt"
  python3 -W ignore experiments/evaluate.py --kind ppo --stoch --wait_min $w --models "results/rl_ppo/ppo_up_lt${tag}_s?.pt"
done
echo "E1 $* 끝"
