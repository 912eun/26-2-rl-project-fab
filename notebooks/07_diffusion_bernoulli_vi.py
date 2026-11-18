"""팀원 리서치의 장비 1대 베르누이 모델을 VI로 풀고 Min/Max 규칙과 비교 (2026-11-13, 확산로 설계 아티팩트용).
모델: 1시간 슬롯, 패밀리 f마다 슬롯당 도착 확률 p_f, 대기열 상한 Q_f, 남은 처리 슬롯 r.
장비가 비었을 때(r=0)만 결정: 기다리기 또는 패밀리 f 시작(q_f >= bmin_f, min(q_f,bmax_f)개 투입).
비용 = 슬롯마다 대기 lot 수 합. 할인 0.995(리서치 설정). 평가는 같은 정책으로 긴 시뮬레이션의 평균 대기 lot 수.
"""
import itertools, sys
import numpy as np

def build(fams, Qmul=3):
    Q = [Qmul * f["bmax"] for f in fams]
    R = max(f["T"] for f in fams)
    shape = tuple(q + 1 for q in Q) + (R + 1,)
    return Q, R, shape

def arrivals(fams, Q, qs):
    """다음 대기열 분포 [(prob, qs')]"""
    out = []
    for bits in itertools.product([0, 1], repeat=len(fams)):
        pr = 1.0; nq = []
        for f, b, q, cap in zip(fams, bits, qs, Q):
            pr *= f["p"] if b else 1 - f["p"]
            nq.append(min(q + b, cap))
        out.append((pr, tuple(nq)))
    return out

def actions(fams, qs):
    acts = [None]
    for i, f in enumerate(fams):
        if qs[i] >= f["bmin"]:
            acts.append(i)
    return acts

def step_after(fams, qs, r, a):
    """행동 뒤 (대기열, 남은 시간). 이번 슬롯 비용은 행동 뒤 대기열 합"""
    qs = list(qs)
    if a is not None:
        k = min(qs[a], fams[a]["bmax"]); qs[a] -= k; r = fams[a]["T"]
    return tuple(qs), r

def solve(fams, gamma=0.995, iters=4000, tol=1e-6):
    Q, R, shape = build(fams)
    V = np.zeros(shape)
    states = list(itertools.product(*[range(s) for s in shape]))
    trans = {}
    for s in states:
        qs, r = s[:-1], s[-1]
        acts = actions(fams, qs) if r == 0 else [None]
        lst = []
        for a in acts:
            q2, r2 = step_after(fams, qs, r, a)
            cost = sum(q2)
            nr = max(r2 - 1, 0)
            lst.append((a, cost, [(p, nq + (nr,)) for p, nq in arrivals(fams, Q, q2)]))
        trans[s] = lst
    for it in range(iters):
        V2 = np.empty_like(V); pol = {}
        for s, lst in trans.items():
            best = None
            for a, c, nx in lst:
                v = c + gamma * sum(p * V[n] for p, n in nx)
                if best is None or v < best - 1e-12:
                    best, ba = v, a
            V2[s] = best; pol[s] = ba
        d = np.abs(V2 - V).max(); V = V2
        if d < tol:
            break
    return V, pol, Q

def rule(name, fams):
    def f(qs, r):
        ok = [i for i in range(len(fams)) if qs[i] >= fams[i]["bmin"]]
        if name == "Max":
            ok = [i for i in range(len(fams)) if qs[i] >= fams[i]["bmax"]]
        if not ok:
            return None
        # 시뮬레이터처럼 차 있는 비율이 큰 패밀리 먼저
        return max(ok, key=lambda i: min(1, qs[i] / fams[i]["bmax"]))
    return f

def simulate(fams, policy, Q, slots=400_000, seed=0):
    rng = np.random.default_rng(seed)
    qs = [0] * len(fams); r = 0; tot = 0; starts = 0; sizes = []
    U = rng.random((slots, len(fams)))
    for t in range(slots):
        if r == 0:
            a = policy(tuple(qs), r)
            if a is not None:
                k = min(qs[a], fams[a]["bmax"]); qs[a] -= k; r = fams[a]["T"]; starts += 1; sizes.append(k / fams[a]["bmax"])
        tot += sum(qs)
        for i, f in enumerate(fams):
            if U[t, i] < f["p"]:
                qs[i] = min(qs[i] + 1, Q[i])
        r = max(r - 1, 0)
    return tot / slots, np.mean(sizes), starts / slots

if __name__ == "__main__":
    # FE_120 LVHM, 리서치 값: 109 (3~4 lot, 6.66h, 0.209/h), 001 (5~6 lot, 8.36h, 0.209/h). 슬롯 1시간이라 처리 7, 8 슬롯으로 반올림
    cases = {
        "FE_120 (리서치, LVHM 장비 1대)": [dict(name="109", p=0.209, bmin=3, bmax=4, T=7), dict(name="001", p=0.209, bmin=5, bmax=6, T=8)],
        "FE_125 (리서치, 패밀리 1개)": [dict(name="004", p=0.333, bmin=3, bmax=4, T=7)],
    }
    for name, fams in cases.items():
        V, pol, Q = solve(fams)
        opt = lambda qs, r: pol[tuple(qs) + (r,)]
        print(name)
        for pn, p in [("Min", rule("Min", fams)), ("Max", rule("Max", fams)), ("VI", opt)]:
            L, fill, sr = simulate(fams, p, Q)
            print(f"  {pn:4s} avg waiting lots {L:6.3f}  mean fill {fill:.3f}  starts/h {sr:.4f}")
        # VI 정책 모양: r=0에서 패밀리별 언제 시작하나
        if len(fams) == 2:
            print("  VI policy at r=0 (rows q109 0..8, cols q001 0..12): . wait, a=109, b=001")
            for q0 in range(0, 9):
                print("   ", q0, "".join({None: ".", 0: "a", 1: "b"}[pol[(q0, q1, 0)]] for q1 in range(0, 13)))
        else:
            print("  VI r=0:", [(q, pol[(q, 0)]) for q in range(0, 9)])
