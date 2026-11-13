"""비교 기준(baseline) 규칙을 작은 환경(공장 도착 기록 재생)에서 돌려 표로 낸다.

규칙 (모두 src/implant_env.py에 있음)
- lssu  : SMT2020/PySCFabSim 기본. 장비가 비면 지금 가스 lot 먼저, 없으면 가장 오래 기다린 lot(hot 우선)의 가스로 바로 교체.
- hold T: lssu와 같되 교체가 필요해지면 T분 기다린 뒤 교체(그사이 지금 가스 lot이 오면 교체 없이 처리).
- cov   : 팀 비교 기준("압력 규칙"). 가스별 압력 = 대기 lot 수 / (그 가스를 맡은 장비 수 + 1).
          가장 높은 압력이 Q 이상이면 그 가스로 교체, 아니면 Tmax분까지 기다린 뒤 가장 오래 기다린 lot의 가스로 교체.
          hot lot이 있거나 장비에 가스가 없으면(PM·고장 뒤) 바로 lssu처럼 교체.
          Q, Tmax는 기록 시드 1에서 튜닝한 값(src/implant_gym.py TUNED_COV,
          results/eda06_fb/tune.json): 128 = (5, 120분), 132 = (3, 240분), 91 = (7, 240분).
          팀원 실험의 압력 규칙(Q=3, Tmax=2시간)은 --q 3 --tmax 120 으로 재현.

평가: 기록 시드 2·3 × 고장 시드 1·2, 240일(앞 60일 버림). RL 시험(experiments/rl_common.py TEST)과 같은 조건.
지표: lot당 지연 D = 대기 + 그 lot이 일으킨 교체 시간(시간), hot lot 지연, 교체 시간 비율, 하루 교체 수.

실행 예:
    python3 baselines/run_baselines.py                       # lssu, hold30, cov(튜닝값)
    python3 baselines/run_baselines.py --q 3 --tmax 120      # cov를 팀원 압력 규칙 값으로
    python3 baselines/run_baselines.py --records 1 --sims 1 2 3
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


def policy(rule, fam, q=None, tmax=None):
    if rule == "lssu":
        return E.lssu
    if rule.startswith("hold"):
        return E.hold(float(rule[4:]) * 60)
    if rule == "cov":
        q0, t0 = G.TUNED_COV[fam]
        return E.thr(q if q is not None else q0, (tmax if tmax is not None else t0) * 60, cov=True)
    raise ValueError(rule)


def job(x):
    fam, rec, ss, rule, q, tmax = x
    return x, E.summarize(G.make_sim(fam, rec, ss, policy(rule, fam, q, tmax)).run())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rules", nargs="+", default=["lssu", "hold30", "cov"])
    p.add_argument("--records", nargs="+", type=int, default=[2, 3], help="도착 기록 시드 (1은 튜닝·학습용)")
    p.add_argument("--sims", nargs="+", type=int, default=[1, 2], help="고장 난수 시드")
    p.add_argument("--q", type=float, default=None, help="cov의 Q를 모든 장비군에 이 값으로")
    p.add_argument("--tmax", type=float, default=None, help="cov의 Tmax(분)를 모든 장비군에 이 값으로")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    jobs = [(f, r, s, rule, a.q, a.tmax) for f in E.FAMILIES for r in a.records for s in a.sims for rule in a.rules]
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for (f, r, s, rule, _, _), m in ex.map(job, jobs):
            res.setdefault((f, rule), []).append(m)
    print(f"기록 시드 {a.records} × 고장 시드 {a.sims}, 240일 (앞 60일 버림). 값은 평균")
    print(f"{'장비군':12s} {'규칙':8s} {'지연D(h)':>8s} {'대기(h)':>8s} {'hot(h)':>7s} {'교체비율':>8s} {'교체/일':>7s}  cov 대비 D")
    for f in E.FAMILIES:
        cov = np.mean([m["delay_h"] for m in res.get((f, "cov"), [])]) if (f, "cov") in res else None
        for rule in a.rules:
            ms = res[(f, rule)]
            g = lambda k: float(np.mean([m[k] for m in ms]))  # noqa: E731
            rel = f"{100 * (g('delay_h') / cov - 1):+.1f}%" if cov else ""
            print(f"{f:12s} {rule:8s} {g('delay_h'):8.2f} {g('wait_h'):8.2f} {g('hot_delay_h'):7.2f} "
                  f"{100 * g('setup_ratio'):7.1f}% {g('switch_per_day'):7.1f}  {rel}")


if __name__ == "__main__":
    main()
