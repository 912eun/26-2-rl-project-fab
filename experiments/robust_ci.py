"""E3 주 결과 구간을 다른 방법으로도 계산해 결론이 같은지 본다 (10장 29~30쪽, 2026-11-20).

시드당 값 = analyze.py와 같은 'cov 대비'(시나리오별 짝 비율 D_RL/D_cov − 1의 평균). 그 값들로
  - t 95% 구간 (analyze.py와 같음, 주 방법)
  - 부트스트랩 95% 구간: 시드 값 N개를 복원 추출해 평균, 10,000번, 가운데 95%
  - IQM(가운데 50% 시드의 평균, Agarwal et al. 2021)과 그 부트스트랩 95% 구간
을 장비군별·전체로 낸다.

예: python3 experiments/robust_ci.py dqn_dqn_up_e3 ppo_ppo_up_lt_e3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402


def iqm(x):
    return float(stats.trim_mean(x, 0.25))


def boot(x, f, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    ms = np.sort([f(rng.choice(x, len(x), replace=True)) for _ in range(B)])
    return float(ms[int(.025 * B)]), float(ms[int(.975 * B) - 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("groups", nargs="+", help="조건 이름 (파일 이름에서 _s<시드>를 뗀 것)")
    a = p.parse_args()
    rules, runs = A.load()
    for g in a.groups:
        stems = sorted(s for s in runs if A.group_of(s) == g)
        print(f"\n[{g}] 시드 {len(stems)}개, cov 대비 % (음수 = cov보다 짧음)")
        print(f"  {'':14s} {'평균':>6s} {'t 95%':>16s} {'부트스트랩 95%':>16s} {'IQM':>6s} {'IQM 부트스트랩':>16s}")
        for fam in A.FAMS + [None]:
            x = np.array([A.per_seed(runs[s], rules, fam)["ratio_cov"] for s in stems]) * 100
            m, lo, hi = A.tci(x)
            blo, bhi = boot(x, np.mean)
            q = iqm(x)
            qlo, qhi = boot(x, iqm)
            name = fam or "전체(lot 가중)"
            print(f"  {name:14s} {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] [{blo:+6.1f}, {bhi:+6.1f}] {q:+6.1f} [{qlo:+6.1f}, {qhi:+6.1f}]")


if __name__ == "__main__":
    main()
