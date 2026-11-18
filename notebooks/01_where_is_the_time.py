# %% [markdown]
# # EDA 01 — 사이클타임은 어디서 쓰이나
#
# 질문: SMT2020(+SMAT2022)에서 RL로 풀 가치가 있는 결정은 어디에 있나?
#
# 1. 정적 용량 분석: 데이터만으로 장비군별 부하율을 계산한다.
# 2. 동적 분해: 지난 프로젝트의 PySCFabSim FIFO 실행(시드 0, 180일)에서
#    사이클타임을 처리, 대기, 반송, 배치 대기로 나누고 대기를 공정 영역별로 배분한다.
#
# 실행: `python notebooks/01_where_is_the_time.py` (VS Code에서는 셀 단위로 실행 가능)
# 출력: results/eda01_*.csv

# %%
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smt2020 import static_capacity  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
RUNS = ROOT / "data" / "processed" / "pyscfabsim_fifo_runs"
pd.set_option("display.width", 200)

# %% [markdown]
# ## 1. 정적 용량 분석
# - `Delay_32`(400대)는 실제 장비가 아니라 공정 간 고정 지연을 표현하는 가상 스테이션이다.
#   이를 빼면 LVHM 913대로 문헌(Kopp et al. WSC 2020)과 일치한다.
# - util_full: 배치를 꽉 채울 때 가용시간 대비 부하율. util_min: 최소 배치로 돌릴 때.

# %%
static = {}
for model in ["SMT2020_LVHM", "SMT2020_HVLM"]:
    g, st, rel = static_capacity(model)
    static[model] = (g, st, rel)
    real = g[g["STNGRP"] != "Delay_32"]
    print(f"== {model}: 장비 {int(real['qty'].sum())}대, 장비군 {len(real)}개, "
          f"주당 투입 {rel['lots_per_week'].sum():.0f} lot")
    area = real.groupby("STNGRP").agg(
        qty=("qty", "sum"), max_util=("util_full", "max"), max_util_minbatch=("util_min", "max"),
        down_frac=("down_frac", "mean")).sort_values("max_util", ascending=False)
    print(area.round(3), "\n")
    real.to_csv(OUT / f"eda01_static_toolgroups_{model}.csv")
    area.to_csv(OUT / f"eda01_static_areas_{model}.csv")

# %% [markdown]
# ## 2. FIFO 실행 결과로 사이클타임 분해
# - lot의 waiting_time은 장비 대기 전체이고, waiting_time_batching은 그중 배치가 차기를 기다린 부분이다.
# - 장비군별 방문당 평균 대기 × lot당 방문 수(투입 비율·샘플링 반영)로 대기를 영역별로 배분한다.
# - 한계: FIFO 규칙, 시드 1개, 1회 실행. 초기 WIP 구간이 포함된다.

# %%
for model in ["SMT2020_LVHM", "SMT2020_HVLM"]:
    g, st, rel = static[model]
    run = json.loads((RUNS / f"greedy_seed0_180days_{model}_fifo.json").read_text())
    lots = pd.DataFrame(run["lots"]).T
    mach = pd.DataFrame(run["machines"]).T

    act = (lots["ACT"] * lots["throughput"]).sum() / lots["throughput"].sum()
    tot = lots[["processing_time", "waiting_time", "transport_time", "waiting_time_batching"]].sum()
    share = tot / tot[["processing_time", "waiting_time", "transport_time"]].sum()

    w = st["lots_per_week"] / rel["lots_per_week"].sum()
    visits = st.assign(v=st["sampling"] * w).groupby("STNFAM")["v"].sum()
    proc = st.assign(p=st["sampling"] * w * st["proc_min"] / 1440).groupby("STNFAM")["p"].sum()
    fam = pd.DataFrame({
        "STNGRP": g["STNGRP"],
        "wait_days_per_lot": visits * mach["waiting_time"].reindex(visits.index).fillna(0),
        "proc_days_per_lot": proc,
        "fifo_util": mach["util"], "static_util": g["util_full"],
    })
    area = fam.groupby("STNGRP")[["wait_days_per_lot", "proc_days_per_lot"]].sum()
    area["wait_share"] = area["wait_days_per_lot"] / area["wait_days_per_lot"].sum()
    area = area.sort_values("wait_share", ascending=False)

    print(f"== {model}: 평균 CT {act:.1f}일 (처리량 {int(lots['throughput'].sum())} lot)")
    print("CT 구성:", {k: f"{v:.1%}" for k, v in share.items()})
    print(area.round(3), "\n")
    fam.sort_values("wait_days_per_lot", ascending=False).to_csv(OUT / f"eda01_fifo_wait_toolgroups_{model}.csv")
    area.to_csv(OUT / f"eda01_fifo_wait_areas_{model}.csv")
