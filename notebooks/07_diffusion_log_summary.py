"""07_diffusion_log_run.py 기록 요약 (2026-11-13, 확산로 설계 아티팩트용).
사용: python notebooks/07_diffusion_log_summary.py RUN_DIR [RUN_DIR ...]
60일 이후만 쓴다(implant 기록과 같은 기준). 시간 단위는 분 → 표에서는 시간.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

WARM = 60 * 1440


def load(d):
    d = Path(d)
    a = pd.read_csv(d / "diff_arrive.csv")
    s = pd.read_csv(d / "diff_start.csv")
    end = max(a.t_min.max(), s.t_min.max())
    return a, s, end


def summarize(d, tools):
    a, s, end = load(d)
    win = end - WARM
    a = a[a.t_min >= WARM]; s = s[s.t_min >= WARM].copy()
    s["waits"] = s.waits_min.astype(str).str.split(";").apply(lambda x: [float(v) for v in x])
    print(f"\n######## {d}  window {win/1440:.1f} days")
    rows = []
    for fam, g in s.groupby("fam"):
        aa = a[a.fam == fam]
        w = np.concatenate(g.waits.values) / 60
        n = tools.get(fam, np.nan)
        rows.append(dict(fam=fam, tools=n, families=aa.groupby(["step", "part"]).ngroups, arrivals=len(aa),
                         arr_per_day=len(aa) / (win / 1440), starts=len(g), lots_started=g.n.sum(),
                         mean_n=g.n.mean(), fill=(g.n / g.bmax).mean(), pct_full=(g.n == g.bmax).mean() * 100,
                         pct_min=(g.n == g.bmin).mean() * 100, wait_h_mean=w.mean(), wait_h_med=np.median(w),
                         busy_h=g.busy_min.mean() / 60, util=g.busy_min.sum() / (n * win) if n else np.nan,
                         free_after_mean=g.free_tools.mean(), pct_free_after=(g.free_tools > 0).mean() * 100))
    T = pd.DataFrame(rows).sort_values("wait_h_mean", ascending=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(T.to_string(index=False))
    allw = np.concatenate(s.waits.values) / 60
    print(f"ALL diffusion: starts {len(s)}, lots {s.n.sum()}, arrivals {len(a)}, mean wait {allw.mean():.2f} h, fill {(s.n/s.bmax).mean():.3f}, pct_min {(s.n==s.bmin).mean()*100:.1f}% pct_full {(s.n==s.bmax).mean()*100:.1f}%")
    return a, s, T


def family_detail(a, s, fam, end):
    print(f"\n--- {fam} per family (step, part) ---")
    aa = a[a.fam == fam]; ss = s[s.fam == fam]
    rows = []
    for (st, pt), g in aa.groupby(["step", "part"]):
        t = np.sort(g.t_min.values)
        ia = np.diff(t) / 60
        cnt = np.histogram(t, bins=np.arange(WARM, end, 480))[0]  # 8시간 창
        cnt1 = np.histogram(t, bins=np.arange(WARM, end, 60))[0]  # 1시간 창
        sg = ss[(ss.step == st) & (ss.part == pt)]
        w = np.concatenate(sg.waits.values) / 60 if len(sg) else np.array([np.nan])
        rows.append(dict(step=st.split("_")[0], part=pt, arrivals=len(t), ia_mean_h=ia.mean(), ia_cv=ia.std() / ia.mean(),
                         disp8h=cnt.var() / cnt.mean(), p_ge2_1h=(cnt1 >= 2).mean() * 100, starts=len(sg),
                         mean_n=sg.n.mean(), bmin=sg.bmin.iloc[0] if len(sg) else np.nan, bmax=sg.bmax.iloc[0] if len(sg) else np.nan,
                         pct_min=(sg.n == sg.bmin).mean() * 100 if len(sg) else np.nan, wait_h=w.mean()))
    T = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(T.to_string(index=False))
    # 공정 단위로 합친 도착(팀원 모델의 패밀리): 1시간 창에서 2개 이상 올 확률
    for st, g in aa.groupby("step"):
        t = np.sort(g.t_min.values)
        cnt1 = np.histogram(t, bins=np.arange(WARM, end, 60))[0]
        cnt8 = np.histogram(t, bins=np.arange(WARM, end, 480))[0]
        print(f"  step {st}: merged products arrivals/h {cnt1.mean():.3f}, P(>=2 in 1h) {(cnt1>=2).mean()*100:.1f}%, disp8h {cnt8.var()/cnt8.mean():.2f}")
    return T


def one_more(a, s, fam=None):
    """bmax 미만으로 시작한 배치: 시작 뒤 같은 패밀리 다음 lot이 올 때까지 걸린 시간"""
    ss = s if fam is None else s[s.fam == fam]
    ss = ss[ss.n < ss.bmax]
    key_t = {k: np.sort(g.t_min.values) for k, g in a.groupby(["fam", "step", "part"])}
    gaps = []
    for r in ss.itertuples():
        t = key_t.get((r.fam, r.step, r.part))
        if t is None:
            continue
        j = np.searchsorted(t, r.t_min, side="right")
        if j < len(t):
            gaps.append((t[j] - r.t_min) / 60)
    g = np.array(gaps)
    if len(g):
        print(f"  one-more-lot gap ({fam or 'all'}): n={len(g)}, mean {g.mean():.2f} h, median {np.median(g):.2f} h, p25 {np.percentile(g,25):.2f}, p75 {np.percentile(g,75):.2f}")
    return g


def compare(dmin, dmax):
    """같은 기간·시드에서 Min 대 Max: 툴셋별 lot당 확산 대기(시간), 배치 채움, 장비 사용 시간"""
    out = {}
    for name, d in [("Min", dmin), ("Max", dmax)]:
        a, s, end = load(d)
        s = s[s.t_min >= WARM].copy()
        s["wsum"] = s.waits_min.astype(str).str.split(";").apply(lambda x: sum(float(v) for v in x))
        g = s.groupby("fam").agg(lots=("n", "sum"), wsum=("wsum", "sum"), fill=("n", lambda x: 0), busy=("busy_min", "sum"))
        g["wait_h"] = g.wsum / g.lots / 60
        g["fill"] = s.assign(f=s.n / s.bmax).groupby("fam").f.mean()
        out[name] = g
        tot = s.wsum.sum() / s.n.sum() / 60
        print(f"{name}: all-diffusion mean wait {tot:.2f} h per lot-visit, lots {s.n.sum()}, fill {(s.n/s.bmax).mean():.3f}")
    T = out["Min"][["wait_h", "fill", "lots"]].join(out["Max"][["wait_h", "fill", "lots"]], lsuffix="_min", rsuffix="_max")
    T["delta_pct"] = (T.wait_h_max / T.wait_h_min - 1) * 100
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(T.to_string())


def revisit(d, fam="Diffusion_FE_120"):
    """같은 lot이 같은 툴셋에 다시 오는 간격: 앞 배치 시작 → 다음 도착"""
    a, s, end = load(d)
    a = a[(a.fam == fam) & (a.t_min >= WARM)]
    s = s[(s.fam == fam)]
    st = {}
    for r in s.itertuples():
        for l in str(r.lots).split(";"):
            st.setdefault(int(l), []).append(r.t_min)
    gaps = []
    for r in a.itertuples():
        prev = [t for t in st.get(r.lot, []) if t < r.t_min]
        if prev:
            gaps.append((r.t_min - max(prev)) / 60)
    g = np.array(gaps)
    print(f"  revisit {fam}: {len(g)} of {len(a)} arrivals had an earlier start in same toolset; gap median {np.median(g)/24:.1f} d, p10 {np.percentile(g,10)/24:.1f} d, min {g.min():.1f} h")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.smt2020 import load_model
    for d in sys.argv[1:]:
        ds = "SMT2020_LVHM" if "LVHM" in d else "SMT2020_HVLM"
        tool = load_model(ds)["tool"]
        tools = {r.STNFAM: int(float(r.STNQTY)) for r in tool.itertuples() if str(r.STNFAM).startswith("Diffusion")}
        a, s, T = summarize(d, tools)
        end = max(a.t_min.max(), s.t_min.max())
        family_detail(a, s, "Diffusion_FE_120", end)
        one_more(a, s); one_more(a, s, "Diffusion_FE_120")
        revisit(d)
    mins = [d for d in sys.argv[1:] if "LVHM" in d and "_Min_" in d]
    maxs = [d for d in sys.argv[1:] if "LVHM" in d and "_Max_" in d]
    if mins and maxs:
        print("\n######## LVHM Min vs Max")
        compare(mins[0], maxs[0])


def floor_wait(d):
    """모이기 하한: 장비가 무한히 많아 최소 배치가 차는 즉시 시작한다고 볼 때의 lot당 대기(시간).
    도착 기록만으로 계산한다. 어떤 시작 규칙도 최소 배치 아래로는 시작할 수 없으므로 이 값보다 짧게 만들 수 없다고 본다."""
    a, s, end = load(d)
    bmin = s.drop_duplicates(["fam", "step", "part"]).set_index(["fam", "step", "part"]).bmin.to_dict()
    a = a[a.t_min >= WARM]
    rows = []
    for fam, g in a.groupby("fam"):
        tot = n = 0
        for (st, pt), x in g.groupby(["step", "part"]):
            t = np.sort(x.t_min.values); k = bmin[(fam, st, pt)]
            m = len(t) // k * k
            for i in range(0, m, k):
                tot += (t[i + k - 1] - t[i:i + k]).sum(); n += k
        rows.append((fam, tot / n / 60))
    return dict(rows)
