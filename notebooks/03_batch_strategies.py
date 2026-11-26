# %% [markdown]
# # EDA 03 — 확산로 배치 시작 규칙이 팹 성과를 얼마나 바꾸나
#
# PySCFabSim-revised에서 배치 전략만 바꾼 2년 실행 결과를 비교한다.
# 실행: `experiments/run_batch_strategies.py` (전략 4종 × HVLM-FIFO, LVHM-CR, 시드 1)
# 통계는 원 논문처럼 2년째에 투입된 lot만 집계한다(1년 warm-up 후 리셋).
#
# 전략 (simulation/greedy.py)
# - Max: 최대 배치가 찰 때까지 기다림 (원판 PySCFabSim 동작)
# - Min: 최소 배치가 되면 바로 투입
# - RoundRobin: Max와 Min을 번갈아
# - Demand: 최소 이상이면 투입, 최대에서 자름
#
# 출력: results/eda03_*.csv

# %%
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "batch_strat"
OUT = ROOT / "results"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

# Kopp/Mönch 기준값 (Heik 외 비교표의 K/P 열, 리서치 에이전트가 Zenodo 15798604에서 읽은 값)
REF = {
    "SMT2020_HVLM": {"Lot_3": 51.21, "Lot_4": 28.38, "HotLot_3": 31.73, "HotLot_4": 18.46, "_throughput": 39479.85},
    "SMT2020_LVHM": {"Lot_1": 47.72, "Lot_5": 21.99, "_throughput": 39573},
}

# %%
rows, lot_rows, mach_rows = [], [], []
for run_dir in sorted(RUNS.glob("*_730d_s*")):
    files = list((run_dir / "greedy").glob("*.json"))
    if not files:
        continue
    dataset, disp, strat = run_dir.name.split("_")[0] + "_" + run_dir.name.split("_")[1], run_dir.name.split("_")[2], run_dir.name.split("_")[3]
    res = json.loads(files[0].read_text())
    lots = pd.DataFrame(res["lots"]).T
    n1 = lots["throughput_one_year"].astype(float)
    regular = ~lots.index.str.contains("Hot")
    for col in ["waiting_time", "waiting_time_batching", "processing_time", "transport_time"]:
        lots[col + "_d"] = lots[col].astype(float) / n1 / 86400
    lots["ontime_pct"] = lots["on_time"].astype(float) / n1 * 100
    lots["dataset"], lots["strategy"] = dataset, strat
    lot_rows.append(lots)

    w = n1[regular]
    act_reg = (lots.loc[regular, "ACT"].astype(float) * w).sum() / w.sum()
    comp = {c: (lots.loc[regular, c + "_d"] * w).sum() / w.sum()
            for c in ["waiting_time", "waiting_time_batching", "processing_time", "transport_time"]}
    mach = pd.DataFrame(res["machines"]).T
    mach["dataset"], mach["strategy"] = dataset, strat
    mach_rows.append(mach)
    diff = mach[mach.index.str.startswith("Diffusion")]
    imp = mach[mach.index.isin(["Implant_128", "Implant_91", "Implant_132"])]
    rows.append(dict(
        dataset=dataset, dispatcher=disp, strategy=strat,
        throughput_total=lots["throughput"].astype(float).sum(),
        throughput_year2=n1.sum(),
        act_regular=act_reg,
        ontime_regular=lots.loc[regular, "on_time"].astype(float).sum() / w.sum() * 100,
        wait_d=comp["waiting_time"], batch_wait_d=comp["waiting_time_batching"],
        proc_d=comp["processing_time"], transport_d=comp["transport_time"],
        diffusion_util_mean=diff["util"].mean(), diffusion_util_max=diff["util"].max(),
        implant_setup_share=imp["setup"].mean(),
        wall_s=(run_dir / "run_info.txt").read_text().split("wall_seconds=")[1].split()[0],
    ))

summary = pd.DataFrame(rows).sort_values(["dataset", "strategy"])
print(summary.round(3).to_string(index=False))
summary.to_csv(OUT / "eda03_batch_strategy_summary.csv", index=False)
if lot_rows:
    pd.concat(lot_rows).to_csv(OUT / "eda03_batch_strategy_lots.csv")
    pd.concat(mach_rows).to_csv(OUT / "eda03_batch_strategy_machines.csv")

# %% 기준값과 비교
for ds, ref in REF.items():
    if not lot_rows:
        break
    L = pd.concat(lot_rows)
    L = L[L["dataset"] == ds]
    print(f"\n== {ds} 기준값 대비 ACT (일)")
    t = L.reset_index().pivot_table(index="index", columns="strategy", values="ACT", aggfunc="first").astype(float)
    t = t.loc[[k for k in ref if not k.startswith("_") and k in t.index]]
    t.insert(0, "ref", [ref[k] for k in t.index])
    print(t.round(1).to_string())
