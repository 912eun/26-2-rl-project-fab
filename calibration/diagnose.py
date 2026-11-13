"""오차 진단: 같은 시드·같은 기간(60~240일), 같은 규칙(lssu)에서 공장 기록과 작은 환경을 여러 지표로 비교한다.

공장 값은 도착 기록 파일(events_v2: arrive / start 행)에서 직접 계산한다. 작은 환경은 같은 도착을 재생하고
lot 투입을 가로채 같은 방식으로 센다. 그래서 두 값은 계산 방식이 같다.

지표 (장비군별)
- wait_h      : lot당 대기(도착 → 투입), 60일 이후 도착
- p50/p90     : 대기 분포
- queue       : 평균 대기 lot 수 (리틀 법칙: 총 대기 시간 / 기간)
- sw/m·d      : 장비 1대 하루 교체 수
- run         : 교체 사이 연속 처리 lot 수 평균, run<7 = 7 lot 미만으로 끝난 연속 처리 비율
- util        : (처리 시간 + 교체 시간) / (장비 수 × 기간). 공장은 기록의 machine_time 사용(PM·고장 시간 제외)

실행: python3 calibration/diagnose.py [--seeds 1 2 3] [--sims 1 .. 10] [--max_gap_h 12] [--fams Implant_132]

결과 (2026-11-13, 고장 시드 10개, max_gap 12h): 대기 재생/공장 − 1 = 128 −1~−5%, 132 +2~+7%, 91 −3~−6%.
공장 값은 9개 모두 재생 10개의 최소~최대 범위 안. 교체 수 −3~0%, 연속 처리 길이 0~+3%, 7 lot 미만 연속 처리 비율,
가동률(처리+교체) 모두 ±4% 안. 즉 남은 차이의 대부분은 고장 난수이고, 모형 차이는 장비군별 ±5% 정도.
max_gap 24h도 비슷(132 +4~+8%, 91 −3~−8%)해서 12h를 유지한다.
"""
import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import implant_env as E  # noqa: E402

T0, T1 = E.WARMUP, 240 * 86400


def metrics(waits, starts, n_m, setup_s):
    """waits: [대기 초] (60일 이후 도착), starts: {장비: [(t, 교체였나, 처리 초)]}"""
    span = T1 - T0
    w = np.array(waits) / 3600
    sw, runs, busy = 0, [], 0.0
    for m, xs in starts.items():
        xs = sorted(x for x in xs if T0 <= x[0] < T1)
        cur = None
        for t, is_sw, mt in xs:
            busy += mt + (setup_s if is_sw else 0)
            if is_sw:
                sw += 1
                if cur is not None:
                    runs.append(cur)
                cur = 1
            elif cur is not None:
                cur += 1
    runs = np.array(runs) if runs else np.array([0])
    return dict(wait_h=w.mean(), p50=np.median(w), p90=np.quantile(w, .9), queue=w.sum() * 3600 / span,
                sw=sw / n_m / (span / 86400), run=runs.mean(), run7=(runs < 7).mean(), util=busy / (n_m * span))


def fab(fam, seed):
    arrive, waits, starts = {}, [], defaultdict(list)
    with E.open_record(E.record_path(seed, "events_v2.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["family"] != fam:
                continue
            key, t = (r["lot"], r["step"]), float(r["t"])
            if r["kind"] == "arrive":
                arrive[key] = t
                continue
            if key in arrive and T0 <= arrive[key] < T1 and t < T1:
                waits.append(t - arrive[key])
            starts[r["machine"]].append((t, r["setup"] != r["machine_setup_before"], float(r["machine_time"])))
    f = E.FAMILIES[fam]
    return metrics(waits, starts, f["n"], f["setup"])


class LogSim(E.Sim):
    def _dispatch(self, m, g):
        st = self.setup if m.gas != g else 0
        lot = (self.qh[g] or self.q[g])[0]
        if not hasattr(self, "dlog"):
            self.dlog = defaultdict(list)
            self.wlog = []
        self.dlog[m.idx].append((self.t, st > 0, lot[1]))
        if T0 <= lot[0] < T1:
            self.wlog.append(self.t - lot[0])
        return super()._dispatch(m, g)


def replay(fam, seed, ss, gap):
    arr = E.load_chains(fam, seed, gap)[0]
    s = LogSim(fam, arr, E.lssu, seed=ss)
    s.run(until=T1)
    f = E.FAMILIES[fam]
    return metrics(s.wlog, s.dlog, f["n"], f["setup"])


def job(x):
    kind, fam, seed, ss, gap = x
    return x, (fab(fam, seed) if kind == "fab" else replay(fam, seed, ss, gap))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--sims", nargs="+", type=int, default=list(range(1, 11)),
                   help="고장 난수 시드. 고장 하나가 평균 10시간이라 실행 1회 대기가 ±10% 흔들린다. 10개 이상 평균을 권장")
    p.add_argument("--max_gap_h", type=float, default=12)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fams", nargs="+", default=list(E.FAMILIES))
    a = p.parse_args()
    gap = a.max_gap_h * 3600
    jobs = [("fab", f, s, 0, gap) for f in a.fams for s in a.seeds]
    jobs += [("rep", f, s, ss, gap) for f in a.fams for s in a.seeds for ss in a.sims]
    res = defaultdict(list)
    with ProcessPoolExecutor(a.workers) as ex:
        for (kind, f, s, ss, _), m in ex.map(job, jobs):
            res[(kind, f)].append(m)
    keys = ["wait_h", "p50", "p90", "queue", "sw", "run", "run7", "util"]
    print(f"lssu, 기록 시드 {a.seeds}, 60~240일. 재생은 고장 시드 {a.sims} 평균. 차이 = 재생/공장 − 1")
    print(f"{'':12s} {'':4s} " + " ".join(f"{k:>8s}" for k in keys))
    for f in a.fams:
        F = {k: np.mean([m[k] for m in res[("fab", f)]]) for k in keys}
        R = {k: np.mean([m[k] for m in res[("rep", f)]]) for k in keys}
        print(f"{f:12s} 공장 " + " ".join(f"{F[k]:8.3f}" for k in keys))
        print(f"{'':12s} 재생 " + " ".join(f"{R[k]:8.3f}" for k in keys))
        print(f"{'':12s} 차이 " + " ".join(f"{100 * (R[k] / F[k] - 1):+7.0f}%" if F[k] else f"{'-':>8s}" for k in keys))


if __name__ == "__main__":
    main()
