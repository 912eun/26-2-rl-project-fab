"""EDA 04-a: implant 가스 셋업 처리 방식 3가지의 2년 결과 비교 (HVLM, FIFO, 배치 Demand, 시드 1).

A nosetup  : 가스 무시            → results/implant_setup/SMT2020_HVLM_fifo_nosetup_Demand_730d_s1
B lssu     : 기본(가스 맞는 lot 먼저) → EDA 03의 results/batch_strat/SMT2020_HVLM_fifo_Demand_730d_s1 재사용
C dedicated: 장비마다 가스 고정     → results/implant_setup/SMT2020_HVLM_fifo_dedicated_Demand_730d_s1
통계는 2년째 투입 lot 기준(1년 뒤 리셋). 출력: results/eda04_implant_modes.csv
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "A_nosetup": ROOT / "results/implant_setup/SMT2020_HVLM_fifo_nosetup_Demand_730d_s1",
    "B_lssu": ROOT / "results/batch_strat/SMT2020_HVLM_fifo_Demand_730d_s1",
    "C_dedicated": ROOT / "results/implant_setup/SMT2020_HVLM_fifo_dedicated_Demand_730d_s1",
}
GAS = ["Implant_128", "Implant_132", "Implant_91"]
pd.set_option("display.width", 220)

rows = []
for mode, d in RUNS.items():
    files = list((d / "greedy").glob("*.json"))
    if not files:
        print("없음:", mode, d)
        continue
    res = json.loads(files[0].read_text())
    lots = pd.DataFrame(res["lots"]).T
    n1 = lots["throughput_one_year"].astype(float)
    reg = ~lots.index.str.contains("Hot")
    w = n1[reg]
    wavg = lambda c: (lots.loc[reg, c].astype(float) * w).sum() / w.sum()  # noqa: E731
    mach = pd.DataFrame(res["machines"]).T
    r = dict(mode=mode, throughput_year2=n1.sum(), act_regular_d=wavg("ACT"),
             ontime_regular=lots.loc[reg, "on_time"].astype(float).sum() / w.sum() * 100,
             wait_d=(lots.loc[reg, "waiting_time"].astype(float)).sum() / w.sum() / 86400)
    for prod in ["Lot_3", "Lot_4", "HotLot_3", "HotLot_4"]:
        if prod in lots.index:
            r[f"ACT_{prod}"] = float(lots.loc[prod, "ACT"])
    for g in GAS:
        r[f"setup_{g}"] = float(mach.loc[g, "setup"])
        r[f"util_{g}"] = float(mach.loc[g, "util"])
    info = d / "implant_info.json"
    if info.exists():
        lw = json.loads(info.read_text())["lot_waiting_at_machine"]
        for g in GAS:
            r[f"wait_h_{g}"] = lw[g][1] / max(lw[g][0], 1) / 3600  # 2년 전체 누적 평균(리셋 없음)
    rows.append(r)

tab = pd.DataFrame(rows).set_index("mode").T
print(tab.round(3).to_string())
tab.to_csv(ROOT / "results" / "eda04_implant_modes.csv")
