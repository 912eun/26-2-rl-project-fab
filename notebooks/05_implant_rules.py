"""EDA 05: implant 단순 개선 규칙이 기본 규칙(lssu)을 이기는가. HVLM, FIFO, 배치 Demand, 2년, 시드 1~3.

규칙: lssu(기준) / longest(대기열이 가장 긴 가스로 교체) / hold15, hold30(교체 전 15·30분 대기)
통계는 2년째 투입 lot 기준(1년 뒤 리셋). 끝난 실행만 집계하므로 중간에 돌려도 된다.
출력: results/eda05_implant_rules_runs.csv (실행별), results/eda05_implant_rules_summary.csv (규칙별)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAS = ["Implant_128", "Implant_132", "Implant_91"]
pd.set_option("display.width", 220)

rows = []
for rule in ["lssu", "longest", "hold15", "hold30"]:
    for seed in [1, 2, 3]:
        d = ROOT / f"results/implant_setup/SMT2020_HVLM_fifo_{rule}_Demand_730d_s{seed}"
        files = list((d / "greedy").glob("*.json"))
        if not (d / "run_info.txt").exists() or not files:
            continue
        res = json.loads(files[0].read_text())
        lots = pd.DataFrame(res["lots"]).T
        n1 = lots["throughput_one_year"].astype(float)
        reg = ~lots.index.str.contains("Hot")
        w = n1[reg]
        wavg = lambda c: (lots.loc[reg, c].astype(float) * w).sum() / w.sum()  # noqa: E731
        mach = pd.DataFrame(res["machines"]).T
        r = dict(rule=rule, seed=seed, throughput_year2=n1.sum(), act_regular_d=wavg("ACT"),
                 ontime_regular=lots.loc[reg, "on_time"].astype(float).sum() / w.sum() * 100)
        for prod in ["Lot_3", "Lot_4", "HotLot_3", "HotLot_4"]:
            if prod in lots.index:
                r[f"ACT_{prod}"] = float(lots.loc[prod, "ACT"])
        info = json.loads((d / "implant_info.json").read_text())
        for g in GAS:
            r[f"setup_{g}"] = float(mach.loc[g, "setup"])
            lw = info["lot_waiting_at_machine"][g]
            r[f"wait_h_{g}"] = lw[1] / max(lw[0], 1) / 3600  # 2년 전체 누적 평균
        if "hold" in info:
            h = info["hold"]
            r["hold_kept_share"] = h["kept_gas"] / max(h["kept_gas"] + h["expired_switch"], 1)
        rows.append(r)

runs = pd.DataFrame(rows)
print(f"끝난 실행 {len(runs)}/12")
runs.to_csv(ROOT / "results/eda05_implant_rules_runs.csv", index=False)
num = [c for c in runs.columns if c not in ("rule", "seed")]
summary = runs.groupby("rule", sort=False)[num].agg(["mean", "std"])
print(summary.T.round(3).to_string())
summary.to_csv(ROOT / "results/eda05_implant_rules_summary.csv")
# 같은 시드끼리 기준(lssu)과의 차이. 난수를 시뮬레이션 전체가 공유하므로 시드 간 변동이 완전히 빠지지는 않는다.
base = runs[runs.rule == "lssu"].set_index("seed")
for rule in runs.rule.unique():
    if rule == "lssu":
        continue
    cur = runs[runs.rule == rule].set_index("seed")
    common = cur.index.intersection(base.index)
    for c in ["act_regular_d", "ontime_regular", "throughput_year2"]:
        diff = (cur.loc[common, c] - base.loc[common, c]).round(2).to_dict()
        print(f"{rule:8s} {c:18s} 시드별 차이 {diff}")
