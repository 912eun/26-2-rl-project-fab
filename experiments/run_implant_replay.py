"""EDA 06: 도착 기록 재생으로 규칙과 rollout을 비교한다. 기준은 docs/eda06_preregistration.md.

단계: validate(환경 검증) / tune(시드 1 튜닝) / test(시드 2, 3 시험) / rollout(미래 정보 비교) / explore(부하·PM 탐색)
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import implant_env as E  # noqa: E402

FEEDBACK = os.environ.get("IMPLANT_FEEDBACK") == "1"  # 재방문 되먹임 판
MAX_GAP = 12 * 3600  # 왕복 12시간 이하 재방문만 연결. 시드 1 검증으로 골랐다
OUT = ROOT / "results" / ("eda06_fb" if FEEDBACK else "eda06")
FAMS = list(E.FAMILIES)
HOLD_T = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
Q = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
TMAX = [15, 30, 60, 120, 240, 480]
ARR = {}


def arr(fam, seed, load=1.0):
    key = (fam, seed)
    if key not in ARR:
        ARR[key] = E.load_chains(fam, seed, MAX_GAP)[0] if FEEDBACK else E.load_arrivals(fam, seed)[0]
    a = ARR[key]
    if load == 1.0:
        return a
    assert not FEEDBACK, "되먹임 판의 부하 조건은 아직 없다"
    return [(x[0] / load, x[1], x[2], x[3]) for x in a]


def flat(fam, seed):
    return E.load_chains(fam, seed, MAX_GAP)[1] if FEEDBACK else arr(fam, seed)


def make(spec):
    k = spec[0]
    if k == "lssu":
        return E.lssu
    if k == "hold":
        return E.hold(spec[1] * 60)
    if k == "thr":
        return E.thr(spec[1], spec[2] * 60)
    if k == "cov":
        return E.thr(spec[1], spec[2] * 60, cov=True)
    raise ValueError(spec)


def one(job):
    fam, seed, spec, load, sim_seed, use_pm = job
    s = E.Sim(fam, arr(fam, seed, load), make(spec), seed=sim_seed, use_pm=use_pm, use_br=use_pm).run()
    return (job, E.summarize(s), E.weekly(s))


def rollout_job(job):
    fam, seed, base, mode, H, load, sim_seed, k = job
    L = 0
    if str(mode).startswith("look"):
        L = float(mode[4:]) * 60
    a = arr(fam, seed, load)
    b = make(base)
    t0 = time.time()
    if mode == "clair":
        pol = E.rollout(b, H=H)
    else:
        pool = [E.make_schedule(E.FAMILIES[fam]["n"], 900 + i) for i in range(5)]
        pol = E.rollout(b, H=H, future=E.past_future(flat(fam, seed), pool, H=H, seed=sim_seed, L=L), k=k)
    s = E.Sim(fam, a, pol, seed=sim_seed).run()
    return (job, E.summarize(s), E.weekly(s), time.time() - t0)


def real_values(fam, seed):
    """같은 시드의 공장 전체 실행에서 잰 값 (60일 이후 도착)."""
    import csv
    arrive, w, n, sw, gas_time = {}, 0.0, 0, 0, E.FAMILIES[fam]["setup"]
    p = E.record_path(seed, "events_v2.csv")
    tl = 0.0
    with E.open_record(p) as fh:
        for r in csv.DictReader(fh):
            if r["family"] != fam:
                continue
            key = (r["lot"], r["step"])
            t = float(r["t"])
            if r["kind"] == "arrive":
                arrive[key] = t
            else:
                tl = max(tl, t)
                if t >= E.WARMUP:
                    if r["setup"] != r["machine_setup_before"]:
                        sw += 1
                    if key in arrive and arrive[key] >= E.WARMUP:
                        w += t - arrive[key]
                        n += 1
    span = tl - E.WARMUP
    return dict(wait_h=w / n / 3600, setup_ratio=sw * gas_time / (E.FAMILIES[fam]["n"] * span), switch_per_day=sw / (span / 86400), n=n)


def ci(diffs, rng=random.Random(0), B=4000):
    n = len(diffs)
    ms = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(B))
    return sum(diffs) / n, ms[int(.025 * B)], ms[int(.975 * B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--H", type=float, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fam", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pool = Pool(a.workers)
    fams = [a.fam] if a.fam else FAMS

    if a.stage == "validate":
        jobs = [(f, s, sp, 1.0, 0, True) for f in fams for s in (1, 2, 3) for sp in (("lssu",), ("hold", 30))]
        t0 = time.time()
        res = list(pool.map(one, jobs))
        print(f"{len(jobs)}회 {time.time()-t0:.1f}초")
        out = []
        for f in fams:
            for s in (1, 2, 3):
                r = real_values(f, s)
                sim = {j[2][0]: m for j, m, _ in res if j[0] == f and j[1] == s}
                row = dict(family=f, seed=s, real=r, lssu=sim["lssu"], hold30=sim["hold"])
                out.append(row)
                l, h = sim["lssu"], sim["hold"]
                print(f"{f} s{s} 대기h 실제 {r['wait_h']:.2f} 재생 {l['wait_h']:.2f} ({l['wait_h']/r['wait_h']-1:+.0%}) | "
                      f"교체 실제 {r['setup_ratio']:.3f} 재생 {l['setup_ratio']:.3f} ({l['setup_ratio']/r['setup_ratio']-1:+.0%}) | "
                      f"hold30: 교체 {h['setup_ratio']/l['setup_ratio']-1:+.0%} 대기 {h['wait_h']/l['wait_h']-1:+.0%} 지연 {h['delay_h']/l['delay_h']-1:+.0%}")
        (OUT / "validate.json").write_text(json.dumps(out, indent=1))

    if a.stage in ("tune", "explore"):
        conds = [(1.0, True)] if a.stage == "tune" else [(1.05, True), (1.10, True), (1.0, False)]
        result = {}
        for load, pm in conds:
            specs = [("lssu",)] + [("hold", t) for t in HOLD_T] + [(k, q, t) for k in ("thr", "cov") for q in Q for t in TMAX]
            jobs = [(f, 1, sp, load, ss, pm) for f in fams for sp in specs for ss in (0, 1)]
            t0 = time.time()
            res = list(pool.map(one, jobs, chunksize=4))
            print(f"[load {load} pm {pm}] {len(jobs)}회 {time.time()-t0:.0f}초")
            agg = {}
            for j, m, _ in res:
                agg.setdefault((j[0], j[2]), []).append(m["delay_h"])
            cond = {}
            for f in fams:
                rows = {sp: sum(v) / len(v) for (ff, sp), v in agg.items() if ff == f}
                best = {}
                for name in ("hold", "thr", "cov"):
                    sp = min((x for x in rows if x[0] == name), key=rows.get)
                    best[name] = sp
                print(f, "lssu", round(rows[("lssu",)], 3), {k: (v[1:], round(rows[v], 3)) for k, v in best.items()})
                cond[f] = dict(lssu=rows[("lssu",)], best={k: list(v) for k, v in best.items()},
                               table={str(k): v for k, v in rows.items()})
            result[f"{load}_{pm}"] = cond
        name = "tune.json" if a.stage == "tune" else "explore_tune.json"
        (OUT / name).write_text(json.dumps(result, indent=1))

    if a.stage in ("test", "explore"):
        tn = json.loads((OUT / ("tune.json" if a.stage == "test" else "explore_tune.json")).read_text())
        final = {}
        for ck, cond in tn.items():
            load, pm = ck.split("_")
            load, pm = float(load), pm == "True"
            for f in fams:
                specs = {"lssu": ("lssu",), **{k: tuple(v) for k, v in cond[f]["best"].items()}}
                jobs = [(f, s, sp, load, ss, pm) for s in (2, 3) for sp in specs.values() for ss in (0, 1)]
                res = list(pool.map(one, jobs))
                m = {}
                wk = {}
                for j, x, w in res:
                    n = next(k for k, v in specs.items() if v == j[2])
                    m.setdefault(n, []).append(x)
                    for kk, vv in w.items():
                        wk.setdefault(n, {})[(j[1], j[4], kk)] = vv
                row = {n: {q: sum(y[q] for y in v) / len(v) for q in v[0]} for n, v in m.items()}
                print(f"\n{ck} {f}", {k: v[1:] for k, v in specs.items() if k != "lssu"})
                for n, v in row.items():
                    print(f"  {n:5s} 지연 {v['delay_h']:.3f} 대기 {v['wait_h']:.3f} 교체 {v['setup_ratio']:.3f} p95 {v['p95_h']:.1f} hot {v['hot_delay_h']:.2f}")
                base = "hold"
                cmp = {}
                for n in ("lssu", "thr", "cov"):
                    keys = [k for k in wk[n] if k in wk[base]]
                    d = [(wk[n][k] - wk[base][k]) / wk[base][k] * 100 for k in keys]
                    mean, lo, hi = ci(d)
                    cmp[n] = (mean, lo, hi)
                    print(f"  {n} − hold: {mean:+.1f}% [{lo:+.1f}, {hi:+.1f}]")
                final[f"{ck}|{f}"] = dict(specs={k: list(v) for k, v in specs.items()}, metrics=row, vs_hold=cmp)
        (OUT / ("test.json" if a.stage == "test" else "explore_test.json")).write_text(json.dumps(final, indent=1))

    if a.stage in ("rollout", "lookahead"):
        tn = json.loads((OUT / "tune.json").read_text())["1.0_True"]
        jobs = []
        for f in fams:
            rows = tn[f]["table"]
            cand = {k: tuple(v) for k, v in tn[f]["best"].items()}
            b = min(cand.values(), key=lambda x: rows[str(x)])
            for s in (2, 3):
                if a.stage == "rollout":
                    jobs.append((f, s, b, "clair", a.H * 3600, 1.0, 0, 1))
                    jobs.append((f, s, b, "online", a.H * 3600, 1.0, 0, a.k))
                else:  # 탐색: 앞으로 L분의 도착을 안다면
                    for L in (0, 30, 60, 120, 240):
                        jobs.append((f, s, b, f"look{L}", a.H * 3600, 1.0, 0, a.k))
        final = []
        for j, m, w, sec in pool.map(rollout_job, jobs):
            print(j[0], "s", j[1], j[3], "base", j[2], {k: round(v, 3) for k, v in m.items() if k in ("delay_h", "wait_h", "setup_ratio", "p95_h")}, f"{sec/60:.1f}분", flush=True)
            base = one((j[0], j[1], j[2], 1.0, 0, True))
            keys = [k for k in w if k in base[2]]
            d = [(w[k] - base[2][k]) / base[2][k] * 100 for k in keys]
            mean, lo, hi = ci(d)
            print(f"   base 지연 {base[1]['delay_h']:.3f} → 차이 {mean:+.1f}% [{lo:+.1f}, {hi:+.1f}]", flush=True)
            final.append(dict(job=list(map(str, j)), metrics=m, base=base[1], diff=(mean, lo, hi), minutes=sec / 60))
            (OUT / f"{a.stage}_H{a.H:g}.json").write_text(json.dumps(final, indent=1))


if __name__ == "__main__":
    main()
