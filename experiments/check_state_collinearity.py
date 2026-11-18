"""상태 후보 23개의 다중공선성 확인 (2026-11-12).

재생 환경(되먹임 판, 시드 기록 1)을 cov와 lssu로 돌리며, RL이 결정할 시점(장비가 비었고 현재 가스가 있고
현재 가스 lot이 없음)마다 상태를 기록한다. 장비군 3개를 합쳐 상관과 VIF를 잰다.
"""
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import implant_env as E  # noqa: E402

COV = {"Implant_128": (5, 120), "Implant_132": (3, 240), "Implant_91": (7, 240)}
NAMES = (["A_n", "A_setup"] + ["B_waited", "B_pm"] +
         [f"C_{g}_{k}" for g in ("g0", "g1", "g2") for k in ("lock", "run", "idle")] + ["C_nogas"] +
         [f"D_{g}_{k}" for g in ("g1", "g2") for k in ("n", "hot", "oldest")] +
         [f"E_{g}_arr1h" for g in ("g0", "g1", "g2")])


class LogDeque(deque):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def append(self, x):
        self.sink.append(x[0])
        super().append(x)


def features(s, m, arr_log):
    n, st, t = len(s.ms), s.setup, s.t
    g0 = m.gas
    others = sorted((g for g in range(s.n_gas) if g != g0), key=lambda g: -s.waiting(g))
    order = [g0] + others
    pos = {g: i for i, g in enumerate(order)}
    c = np.zeros(10)
    for o in s.ms:
        if o is m:
            continue
        if o.gas < 0:
            c[9] += 1
        elif not o.free:
            c[pos[o.gas] * 3 + (0 if o.left is not None else 1)] += 1
        else:
            c[pos[o.gas] * 3 + 2] += 1
    c /= (n - 1)
    d = []
    for g in others:
        firsts = [x[0][0] for x in (s.q[g], s.qh[g]) if x]
        d += [len(s.q[g]) / n, len(s.qh[g]), (t - min(firsts)) / st if firsts else 0.0]
    e = []
    for g in order:
        lg = arr_log[g]
        while lg and lg[0] < t - 3600:
            lg.popleft()
        e.append(len(lg) / n)
    pm = min(p for p in m.pieces) / E.PIECES / E.MIN_RUN
    return [n / 10, st / 3600, (t - m.hold_since) / st, pm, *c, *d, *e]


def run(fam, base, seed=1, p_rec=0.2):
    rows, rng = [], random.Random(0)
    q, tmax = COV[fam]
    pol = E.thr(q, tmax * 60, cov=True) if base == "cov" else E.lssu
    arrivals = E.load_chains(fam, seed, 12 * 3600)[0]
    sim = E.Sim(fam, arrivals, None, seed=seed)
    sink = [deque() for _ in range(sim.n_gas)]
    sim.q = [LogDeque(sink[g]) for g in range(sim.n_gas)]
    sim.qh = [LogDeque(sink[g]) for g in range(sim.n_gas)]
    last = {}

    def policy(s, m):
        # 같은 장비가 기다리는 동안 매 사건마다 불리므로, 장비별로 시각이 바뀐 호출만 표본으로 쓴다
        if m.gas >= 0 and s.t >= E.WARMUP and last.get(m.idx) != s.t and rng.random() < p_rec:
            rows.append(features(s, m, sink))
        last[m.idx] = s.t
        return pol(s, m)
    sim.policy = policy
    sim.run()
    return rows


def vif(X):
    C = np.corrcoef(X, rowvar=False)
    return np.diag(np.linalg.pinv(C)), C


if __name__ == "__main__":
    allx = []
    for base in ("cov", "lssu"):
        for fam in E.FAMILIES:
            r = run(fam, base)
            print(base, fam, len(r), "표본", flush=True)
            allx += r
    X = np.array(allx)
    np.save(ROOT / "results" / "state_features_s1.npy", X)
    sd = X.std(0)
    keep = sd > 1e-9
    print("\n변동 없는 변수:", [n for n, k in zip(NAMES, keep) if not k])
    names = [n for n, k in zip(NAMES, keep) if k]
    Xk = X[:, keep]
    v, C = vif(Xk)
    print("\n== VIF (전체 23개) ==")
    for n_, x in sorted(zip(names, v), key=lambda z: -z[1]):
        print(f"{n_:14s} {x:10.1f}")
    print("\n== |상관| > 0.7 쌍 ==")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(C[i, j]) > 0.7:
                print(f"{names[i]:14s} {names[j]:14s} {C[i, j]:+.2f}")
    # C 합 = 1 함정을 풀기 위해 C_nogas를 빼고 다시
    drop = {"C_nogas"}
    k2 = [i for i, n_ in enumerate(names) if n_ not in drop]
    v2, _ = vif(Xk[:, k2])
    print("\n== VIF (C_nogas 제외) ==")
    for n_, x in sorted(zip([names[i] for i in k2], v2), key=lambda z: -z[1]):
        print(f"{n_:14s} {x:10.1f}")
