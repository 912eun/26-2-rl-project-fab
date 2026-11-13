"""검증 점수의 잡음 크기를 잰다: 같은 정책(cov를 RL 자리에 넣은 것)을 검증 구간에 넣고, 고장 시드 묶음만 바꿔 점수가 얼마나 흔들리나.

검증 점수 = 구간별 (정책 비용 ÷ 같은 고장 시드의 cov 비용)의 평균(짝 비교). 여기서는 검증 구간의 고장 시드 묶음을
K개로 바꿔 가며(정책과 cov를 같이 바꿈) 같은 정책의 점수가 얼마나 흔들리나 본다.
또 cov를 RL 자리에 넣은 점수(기본 고장 시드) 자체가 이 검증의 기준점이다(옛 검증에서는 0.952).

실행: python3 experiments/val_noise.py [--K 8]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import implant_gym as G  # noqa: E402
import rl_common as C  # noqa: E402


class TeacherEnv(G.ImplantEnv):
    """RL 차례마다 cov가 고를 행동을 self.teacher에 남긴다."""

    def _policy(self, s, m):
        if s.t < self.t_start or not G.rl_turn(s, m):
            return self.base(s, m)
        obs, gs = G.observe(s, m, self.use_upstream)
        mask = np.array([True, s.waiting(gs[1]) > 0, s.waiting(gs[2]) > 0])
        g, _ = self.base(s, m)
        self.teacher = 0 if g is None or g < 0 else gs.index(g)
        a = self.parent.switch((obs, mask))
        return (None, None) if a == 0 else (gs[a], None)


def cov_job(x):
    (fam, day, ss), off = x
    return C._window_cost_cov((fam, day, ss + off))


def job(x):
    (fam, day, ss), off = x
    env = TeacherEnv(seeds=(C.VAL_REC,))
    env.reset(options=dict(fam=fam, seed=C.VAL_REC, sim_seed=ss + off, t_start=day * 86400, t_end=(day + C.VAL_DAYS) * 86400))
    cost = 0.0
    while True:
        _, r, _, done, _ = env.step(env.teacher)
        cost -= r
        if done:
            return cost


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        scores = []
        for k in range(a.K):
            base = list(ex.map(cov_job, [(w, 100 * k) for w in C.VAL]))
            costs = list(ex.map(job, [(w, 100 * k) for w in C.VAL]))
            scores.append(float(np.mean([c / b for c, b in zip(costs, base)])))
            print(f"고장 시드 묶음 {k}: 검증 점수 {scores[-1]:.3f}", flush=True)
    s = np.array(scores)
    print(f"\ncov를 RL 자리에 넣은 검증 점수 (기본 묶음 0) = {s[0]:.3f}")
    print(f"묶음 {a.K}개: 평균 {s.mean():.3f}, 표준편차 {s.std(ddof=1):.3f}, 범위 {s.min():.3f}~{s.max():.3f}")
    print(f"두 설정 비교 시 잡음 기준(두 점수 차이의 표준편차 ≈ √2 × sd) = {np.sqrt(2) * s.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
