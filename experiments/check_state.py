"""상태 설계 점검 세 가지 (2026-11-12).

1. 빠진 정보: 결정 시점에 "교체 시간 안에 현재 가스 lot이 오나(기다리면 교체를 피하나)"를 상태로 얼마나 맞히나.
   상태만 / + 처리 중 장비가 끝나는 시각 / + 앞 공정 lot 수 를 비교한다 (공장 기록).
2. 스케일: 장비군별 변수 범위.
3. 의미 차이: 같은 규칙(cov, 공장 파라미터)으로 재생 환경과 공장에서 뽑은 상태 분포 비교.

입력: 공장 `--log_state` 실행 결과 (results/implant_setup/SMT2020_HVLM_fifo_cov_Demand_<days>d_s1/state_log.csv)
실행: python3 experiments/check_state.py [days]
"""
import bisect
import csv
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import implant_env as E  # noqa: E402
import implant_state as S  # noqa: E402

FACTORY_COV = {"Implant_128": (5, 240), "Implant_132": (3, 240), "Implant_91": (5, 480)}
OUT = ROOT / "results" / "state_check"


class LogDeque(deque):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def append(self, x):
        self.sink.append(x[0])
        super().append(x)


def replay_rows(fam, seed=1, p_rec=0.3):
    """재생 환경을 cov(공장 파라미터)로 돌리며 첫 결정 시점의 상태를 기록한다."""
    q_, tmax = FACTORY_COV[fam]
    base = E.thr(q_, tmax * 60, cov=True)
    sim = E.Sim(fam, E.load_chains(fam, seed, 12 * 3600)[0], None, seed=seed)
    arr_all = [[] for _ in range(sim.n_gas)]
    sim.q = [LogDeque(arr_all[g]) for g in range(sim.n_gas)]
    sim.qh = [LogDeque(arr_all[g]) for g in range(sim.n_gas)]
    rng, rows = random.Random(12345), []

    def policy(s, m):
        if m.gas >= 0 and m.hold_since == s.t and s.t >= E.WARMUP and rng.random() < p_rec:
            ends = {}
            for e in s.ev:
                if e[2] == E.DONE and e[4] == s.busy_id[e[3]]:
                    ends[e[3]] = e[0]
            others = [(o.gas if o.gas >= 0 else None, not o.free, o.left is not None, ends.get(o.idx))
                      for o in s.ms if o is not m]
            queue = {}
            for g in range(s.n_gas):
                firsts = [x[0][0] for x in (s.q[g], s.qh[g]) if x]
                queue[g] = (len(s.q[g]), len(s.qh[g]), min(firsts) if firsts else None)
            arr1h = {}
            for g in range(s.n_gas):
                a = arr_all[g]
                arr1h[g] = len(a) - bisect.bisect_left(a, s.t - 3600)
            v = S.View(n=len(s.ms), setup_s=s.setup, now=s.t, gases=list(range(s.n_gas)), gas0=m.gas,
                       waited_s=s.t - m.hold_since, pm_pieces=min(m.pieces), others=others, queue=queue, arr1h=arr1h)
            up = s.upstream()
            gs = S.order(v)
            u = [up[0][g] / v.n for g in gs] + [up[1][g] / v.n for g in gs]
            rows.append((s.t, m.gas, S.encode(v), S.extras(v), u))
        return base(s, m)
    sim.policy = policy
    sim.run()
    return rows, arr_all


def label(t, gas0, arrivals_by_gas, horizon):
    a = arrivals_by_gas[gas0]
    i = bisect.bisect_right(a, t)
    return int(i < len(a) and a[i] <= t + horizon)


def load_factory(days):
    d = ROOT / f"results/implant_setup/SMT2020_HVLM_fifo_cov_Demand_{days}d_s1"
    arr = {}
    with open(d / "state_arrivals.csv") as fh:
        for r in csv.DictReader(fh):
            arr.setdefault(r["family"], {}).setdefault(r["gas"], []).append(float(r["t"]))
    for f in arr.values():
        for v in f.values():
            v.sort()
    rows = list(csv.DictReader(open(d / "state_log.csv")))
    return rows, arr


def auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p))
    r[order] = np.arange(1, len(p) + 1)
    # 동점 평균 순위
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_auc(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
    return auc(yte, m.predict_proba(Xte)[:, 1])


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b))))


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    OUT.mkdir(parents=True, exist_ok=True)
    frows, farr = load_factory(days)
    names, xn = S.NAMES, S.EXTRA_NAMES
    upn = ["U1_g0", "U1_g1", "U1_g2", "U2_g0", "U2_g1", "U2_g2"]
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"공장 기록: {len(frows)} 결정 (cov, 시드 1, {days}일, 60일 이후, 30% 표본)")
    say(f"잠금 상태에서 기록된 결정 (0이어야 함): {sum(int(r['locked']) for r in frows)}")

    # ---------- 1. 빠진 정보 ----------
    say("\n## 1. 빠진 정보: 교체 시간 안에 현재 가스 lot이 오나를 맞히는 정도 (AUC, 0.5=동전, 1=완벽)")
    say("앞 70% 기간으로 학습, 뒤 30%로 시험. 장비군 3개를 합친 한 모델.")
    fam_setup = {f: E.FAMILIES[f]["setup"] for f in E.FAMILIES}
    F = {k: np.array([[float(r[c]) for c in cols] for r in frows]) for k, cols in
         (("base", names), ("busy", xn), ("up", upn))}
    t = np.array([float(r["t"]) for r in frows])
    y = np.array([label(float(r["t"]), r["gas0"], farr[r["family"]], fam_setup[r["family"]]) for r in frows])
    cut = np.quantile(t, 0.7)
    tr, te = t < cut, t >= cut
    say(f"기다리면 교체 시간 안에 현재 가스 lot이 오는 비율: {y.mean():.1%}")
    for h in (10, 20, 30):
        yh = np.array([label(float(r["t"]), r["gas0"], farr[r["family"]], h * 60) for r in frows])
        a0 = fit_auc(F["base"][tr], yh[tr], F["base"][te], yh[te])
        Xu = np.hstack([F["base"], F["up"]])
        a1 = fit_auc(Xu[tr], yh[tr], Xu[te], yh[te])
        Xb = np.hstack([F["base"], F["busy"]])
        a2 = fit_auc(Xb[tr], yh[tr], Xb[te], yh[te])
        say(f"  {h}분 안에 (실제 비율 {yh.mean():.0%}): 상태 {a0:.3f} / + 끝나는 시각 {a2:.3f} / + 앞 공정 {a1:.3f}")
    say("  아래는 교체 시간 안에 오나(거의 항상 1이라 참고용)")
    sets = {"상태 23개": ["base"], "+ 처리 중 장비가 끝나는 시각": ["base", "busy"],
            "+ 앞 공정 lot 수": ["base", "up"], "+ 둘 다": ["base", "busy", "up"]}
    res1 = {}
    for k, parts in sets.items():
        X = np.hstack([F[p] for p in parts])
        res1[k] = fit_auc(X[tr], y[tr], X[te], y[te])
        say(f"  {k:28s} AUC {res1[k]:.3f}")
    # 앞 공정 정보만으로 얼마나 맞히나 (참고)
    say(f"  {'(참고) 앞 공정 lot 수만':28s} AUC {fit_auc(F['up'][tr], y[tr], F['up'][te], y[te]):.3f}")

    # ---------- 2. 스케일 ----------
    say("\n## 2. 스케일: 장비군별 중앙값 / 95% 지점 / 최대 (공장 기록)")
    fam = np.array([r["family"] for r in frows])
    hdr = f"{'변수':14s}" + "".join(f"{f:>26s}" for f in E.FAMILIES)
    say(hdr)
    scale_rows = []
    for j, n_ in enumerate(names):
        cells = []
        for f in E.FAMILIES:
            x = F["base"][fam == f, j]
            q = (np.median(x), np.quantile(x, 0.95), x.max())
            cells.append(q)
        scale_rows.append((n_, cells))
        say(f"{n_:14s}" + "".join(f"{c[0]:8.2f}/{c[1]:7.2f}/{c[2]:7.2f}  " for c in cells))

    # ---------- 3. 재생 vs 공장 ----------
    say("\n## 3. 의미 차이: 같은 cov 규칙에서 재생 환경 vs 공장 (시드 1)")
    say("값: 평균(재생) / 평균(공장), KS 거리 (0=같은 분포, 1=완전히 다름). KS>0.2를 표시한다.")
    flagged = []
    for f in E.FAMILIES:
        rr, rarr = replay_rows(f)
        R = np.array([x[2] for x in rr])
        Fb = F["base"][fam == f]
        say(f"\n[{f}] 결정 수 재생 {len(R)}, 공장 {len(Fb)} (재생 240일 기록 / 공장 {days}일)")
        for j, n_ in enumerate(names):
            if n_ in ("A_n", "A_setup_h", "B_waited"):
                continue
            k = ks(R[:, j], Fb[:, j])
            mark = " ←" if k > 0.2 else ""
            if mark:
                flagged.append((f, n_, k))
            say(f"  {n_:14s} {R[:, j].mean():7.3f} / {Fb[:, j].mean():7.3f}   KS {k:.2f}{mark}")
        RU = np.array([x[4] for x in rr])
        FU = F["up"][fam == f]
        for j, n_ in enumerate(upn):
            k = ks(RU[:, j], FU[:, j])
            mark = " ←" if k > 0.2 else ""
            if mark:
                flagged.append((f, n_, k))
            say(f"  {n_:14s} {RU[:, j].mean():7.3f} / {FU[:, j].mean():7.3f}   KS {k:.2f}{mark}")
        for h in (10, 30):
            yh = np.array([label(x[0], x[1], rarr, h * 60) for x in rr])
            Rb = R
            cutr = np.quantile([x[0] for x in rr], 0.7)
            trr = np.array([x[0] < cutr for x in rr])
            a0 = fit_auc(Rb[trr], yh[trr], Rb[~trr], yh[~trr])
            a1 = fit_auc(np.hstack([Rb, RU])[trr], yh[trr], np.hstack([Rb, RU])[~trr], yh[~trr])
            say(f"  (재생) {h}분 안에 현재 가스 lot 도착 AUC: 상태 {a0:.3f} → + 앞 공정 {a1:.3f}")
        yr = np.array([label(x[0], x[1], rarr, E.FAMILIES[f]["setup"]) for x in rr])
        yf = y[fam == f]
        say(f"  (라벨) 기다리면 lot이 오는 비율 재생 {yr.mean():.1%} / 공장 {yf.mean():.1%}")
    say("\nKS>0.2: " + (", ".join(f"{a} {b} {c:.2f}" for a, b, c in flagged) if flagged else "없음"))
    (OUT / f"report_{days}d.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
