"""E1-0 사전 분석 (E1 설계 문서): 결정 순간부터 '지금 가스' lot이 다음에 도착하기까지의 시간 분포. 학습 없음.

- 재생 환경을 기록 시드 1(학습용)에서 cov(기다리기 = 다음 사건까지)로 돌린다. 검증(7)·시험(2·3) 기록은 쓰지 않는다.
- 결정 순간 = 정책이 불리는 순간(장비가 비었고, 가스가 있고, 지금 가스 lot이 없고, 교체 후보가 있을 때) 중
  hot lot이 없고(RL 차례) 60~240일인 때.
- 그 시각 이후 같은 장비군에 같은 가스 lot(hot 포함)이 처음 도착하는 시각까지의 간격을 잰다. 도착 후보는 기록된 도착과
  재방문 연결로 잡힌 도착 예정 사건 둘 다 본다.
- 장비군별·전체 25/50/75/90% 분위수, 결정 수, 그리고 후보 길이(5/10/20/40/80분) 안에 오는 비율.

실행: python3 experiments/e1_prewait.py [--sims 1 2]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import implant_env as E  # noqa: E402
import implant_gym as G  # noqa: E402

T1 = 240 * 86400
CANDS = [5, 10, 20, 40, 80]


def next_arrival(s, g):
    t = float("inf")
    for a in s.arr[s.i:]:  # 기록된 도착 (시간순)
        if a[1] == g:
            t = a[0]
            break
    for e in s.ev:  # 재방문 연결로 잡힌 도착 예정
        if e[2] == E.ARRIVE and e[4][1] == g and e[0] < t:
            t = e[0]
    return t


def job(x):
    fam, ss = x
    q, tm = G.TUNED_COV[fam]
    base = E.thr(q, tm * 60, cov=True)
    gaps = []

    def pol(s, m):
        if E.WARMUP <= s.t < T1 and G.rl_turn(s, m):
            gaps.append((next_arrival(s, m.gas) - s.t) / 60)
        return base(s, m)
    G.make_sim(fam, 1, ss, pol).run()
    return fam, gaps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sims", nargs="+", type=int, default=[1, 2])
    a = p.parse_args()
    res = {}
    with ProcessPoolExecutor(6) as ex:
        for fam, gaps in ex.map(job, [(f, s) for f in E.FAMILIES for s in a.sims]):
            res.setdefault(fam, []).extend(gaps)
    res["전체"] = [x for f in E.FAMILIES for x in res[f]]
    print(f"기록 시드 1, cov, 고장 시드 {a.sims}, 60~240일. 결정 순간 → 지금 가스 lot 다음 도착(분)")
    print(f"{'장비군':12s} {'25%':>6s} {'50%':>6s} {'75%':>6s} {'90%':>6s} {'결정 수':>8s}  " + " ".join(f"≤{c}분" for c in CANDS))
    for k, v in res.items():
        v = np.array(v)
        fin = v[np.isfinite(v)]
        qs = np.quantile(fin, [.25, .5, .75, .9])
        share = " ".join(f"{100 * (v <= c).mean():5.0f}%" for c in CANDS)
        print(f"{k:12s} " + " ".join(f"{x:6.1f}" for x in qs) + f" {len(v):8d}  {share}")


if __name__ == "__main__":
    main()
