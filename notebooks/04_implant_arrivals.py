"""EDA 04-b: implant 가스 장비군에 lot이 어떻게 들어오고, 기본 규칙(LSSU)이 언제 가스를 바꾸는지 본다.

입력: results/implant_setup/SMT2020_HVLM_fifo_lssu_Demand_240d_s1/events.csv
      (experiments/run_implant_setup.py --mode lssu --days 240 --log_events)
앞 60일은 공장이 채워지는 구간이라 뺀다.

보는 것
1. 가스별 도착률과 몰림 정도: 1시간·8시간 단위 도착 수의 분산/평균(포아송이면 1, 1보다 크면 몰려서 옴)
2. 가스별 대기열 길이와 대기 시간 분포(평균, 95% 지점)
3. 가스 교체: 하루 몇 번, 교체 순간에 원래 가스 대기열이 비어 있었는지(=교체가 강제였는지)
4. 결정 순간 수: 하루에 '어느 가스를 돌릴지' 고를 일이 몇 번 생기는지
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "implant_setup" / "SMT2020_HVLM_fifo_lssu_Demand_240d_s1"
OUT = ROOT / "results"
WARM = 60 * 86400

ev = pd.read_csv(RUN / "events.csv", keep_default_na=False)
ev["t"] = ev["t"].astype(float)
arr = ev[ev.kind == "arrive"].copy()
st = ev[ev.kind == "start"].copy()
# lot은 한 번에 한 공정에만 있으므로, 같은 lot의 기록은 도착 → 시작 → 도착 → 시작 순서로 번갈아 나온다.
# 시작 기록은 바로 앞의 같은 장비군 도착 기록과 짝짓는다(초기 WIP는 도착 기록이 없으니 짝이 없다).
e2 = ev.sort_values(["lot", "t", "kind"]).reset_index(drop=True)  # 같은 시각이면 arrive가 start보다 앞
prev = e2.groupby("lot").shift(1)
m = (e2.kind == "start") & (prev.kind == "arrive") & (prev.family == e2.family)
pairs = pd.DataFrame({"lot": e2.lot[m], "family": e2.family[m], "setup": e2.setup[m], "t_a": prev.t[m],
                      "t_s": e2.t[m], "machine": e2.machine[m], "machine_setup_before": e2.machine_setup_before[m]})
# 아직 시작 못 한 도착(분석 끝 시점 대기 중)도 대기열 계산에 넣는다
nxt = e2.groupby("lot").shift(-1)
open_ = (e2.kind == "arrive") & ~((nxt.kind == "start") & (nxt.family == e2.family))
pairs = pd.concat([pairs, pd.DataFrame({"lot": e2.lot[open_], "family": e2.family[open_], "setup": e2.setup[open_],
                                        "t_a": e2.t[open_], "t_s": np.nan})], ignore_index=True)
pairs["wait_h"] = (pairs["t_s"] - pairs["t_a"]) / 3600
bad = (pairs.wait_h < 0).mean()
pairs = pairs[pairs.t_a >= WARM]

days = (ev.t.max() - WARM) / 86400
rows = []
for (fam, gas), g in arr[arr.t >= WARM].groupby(["family", "setup"]):
    t = g.t.values
    r = {"family": fam, "gas": gas, "arrivals_per_day": len(t) / days}
    for name, w in (("1h", 3600), ("8h", 8 * 3600)):
        bins = np.arange(WARM, ev.t.max() + w, w)
        c, _ = np.histogram(t, bins)
        r[f"dispersion_{name}"] = c.var() / c.mean()  # 포아송이면 1
        r[f"zero_share_{name}"] = (c == 0).mean()
    pw = pairs[(pairs.family == fam) & (pairs.setup == gas)].wait_h
    r["wait_mean_h"], r["wait_p95_h"], r["wait_max_h"] = pw.mean(), pw.quantile(.95), pw.max()
    rows.append(r)
gas_tab = pd.DataFrame(rows)

# 대기열 길이(가스별) 시계열: 도착 +1, 시작 −1. 시작 쪽은 짝지어진 것만 써야 일관된다.
qrows = []
for (fam, gas), g in pairs.groupby(["family", "setup"]):
    e = pd.concat([pd.DataFrame({"t": g.t_a, "d": 1}), pd.DataFrame({"t": g.t_s.dropna(), "d": -1})]).sort_values("t")
    e["q"] = e.d.cumsum()
    dt = np.diff(np.append(e.t.values, e.t.values[-1]))
    q = e.q.values
    qrows.append({"family": fam, "gas": gas, "queue_mean": np.average(q, weights=dt),
                  "queue_p95": np.quantile(np.repeat(q, np.maximum(1, (dt / 600).astype(int))), .95),
                  "queue_zero_time_share": dt[q == 0].sum() / dt.sum()})
gas_tab = gas_tab.merge(pd.DataFrame(qrows), on=["family", "gas"])

# 가스 교체: 시작 시점에 장비의 기존 가스와 다른 가스로 시작한 경우 (정비 뒤 '' → 가스는 따로 센다)
s = st[st.t >= WARM].copy()
s["switch"] = (s.machine_setup_before != s.setup) & (s.machine_setup_before != "")
s["after_pm"] = s.machine_setup_before == ""
# 교체 순간, 장비의 원래 가스 대기열 길이(그 장비군 전체 기준)
pairs_all = pairs.copy()


def queue_at(fam, gas, t):
    g = pairs_all[(pairs_all.family == fam) & (pairs_all.setup == gas)]
    return int(((g.t_a <= t) & ((g.t_s > t) | g.t_s.isna())).sum())


sw = s[s.switch]
sample = sw.sample(min(400, len(sw)), random_state=0) if len(sw) else sw
old_q = [queue_at(r.family, r.machine_setup_before, r.t) for r in sample.itertuples()]
sw_rows = []
for fam, g in s.groupby("family"):
    swf = g[g.switch]
    oq = [q for q, f in zip(old_q, sample.family) if f == fam]
    sw_rows.append({"family": fam, "starts_per_day": len(g) / days, "switches_per_day": len(swf) / days,
                    "after_pm_setups_per_day": g.after_pm.sum() / days,
                    "switch_with_old_gas_queue_empty": np.mean([q == 0 for q in oq]) if oq else np.nan,
                    "old_gas_queue_at_switch_mean": np.mean(oq) if oq else np.nan})
sw_tab = pd.DataFrame(sw_rows)

pd.set_option("display.width", 200)
print(f"분석 구간 {days:.0f}일 (60일 이후). 음수 대기(짝짓기 오류) 비율 {bad:.3%}\n")
print("== 가스별 도착·대기\n", gas_tab.round(3).to_string(index=False))
print("\n== 장비군별 투입·가스 교체\n", sw_tab.round(3).to_string(index=False))
gas_tab.to_csv(OUT / "eda04_implant_gas_arrivals.csv", index=False)
sw_tab.to_csv(OUT / "eda04_implant_switches.csv", index=False)
