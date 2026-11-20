"""무작위 정책을 시험 시나리오에서 평가한다 (10장 22쪽 "첫 확인: 무작위 정책과 비교", 2026-11-20).

RL 차례(rl_turn)마다 마스크로 허용된 행동(기다리기 / 대기 lot이 있는 가스로 교체) 중 하나를 같은 확률로 고른다.
hot lot·가스 없음 같은 나머지 결정은 RL 정책과 똑같이 cov가 처리한다. 기다리기는 --wait_min분(기본 40, 최종 설정).
난수 시드마다 결과 파일 하나: results/eval/random_random_s<시드>.json (evaluate.py와 같은 형식이라 analyze.py가 그대로 읽는다).

예: python3 experiments/eval_random.py --seeds 0 1 2
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import implant_gym as G  # noqa: E402
import rl_common as C  # noqa: E402

OUT = ROOT / "results" / "eval"


def job(x):
    seed, fam, rec, ss, wait_min = x
    rng = np.random.default_rng([seed, rec, ss, C.TEST.index((fam, rec, ss))])

    def act(obs, mask):
        return int(rng.choice(np.flatnonzero(mask)))
    return x, G.run_policy(fam, rec, ss, act, True, wait_min)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--wait_min", type=float, default=40)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for seed in a.seeds:
            jobs = [(seed, f, r, ss, a.wait_min) for f, r, ss in C.TEST]
            res = {f"{x[1]}|{x[2]}|{x[3]}": m for x, m in ex.map(job, jobs)}
            out = OUT / f"random_random_s{seed}.json"
            out.write_text(json.dumps(dict(kind="random", model="uniform over masked actions", use_upstream=True,
                                           stoch=True, wait_min=a.wait_min, scenarios=res)))
            print("저장", out.name, flush=True)


if __name__ == "__main__":
    main()
