# %% [markdown]
# # EDA 02 — 셋업을 넣으면 부하가 어떻게 바뀌나
#
# EDA 01의 정적 부하에는 셋업이 빠져 있었다. 셋업이 있는 장비군만 골라 세 시나리오로 다시 계산한다.
# - no_setup: 셋업 0 (EDA 01과 같음)
# - campaign: 최소 연속 처리 수(MINRUN, implant 가스는 7 lot)마다 한 번 셋업. MINRUN이 없으면 방문마다
# - worst: 방문마다 셋업
# 그리고 셋업 종류별로 필요한 장비 대수를 계산해, 장비를 셋업별로 전용하면 셋업을 피할 수 있는지 본다.
# 마지막으로 FIFO 실행(시드 0, 180일)에서 실제 셋업 시간 비율을 확인한다.
#
# 실행: `python notebooks/02_setup_load.py` → results/eda02_*.csv

# %%
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smt2020 import WEEK_MIN, load_model, static_capacity, to_min  # noqa: E402

OUT = ROOT / "results"
RUNS = ROOT / "data" / "processed" / "pyscfabsim_fifo_runs"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)

# %%
for model in ["SMT2020_LVHM", "SMT2020_HVLM"]:
    d = load_model(model)
    g, st, _ = static_capacity(model)
    r = d["route"]
    st = st.copy()
    st["SETUP"] = r["SETUP"].values
    st["STIME"] = [to_min(float(a), u) if pd.notna(a) else np.nan for a, u in zip(r["STIME"], r["STUNITS"])]
    stab = d["setup"].copy()
    stab["t"] = [to_min(float(a), u) for a, u in zip(stab["STIME"], stab["STUNITS"])]
    st["su_time"] = st["STIME"].fillna(st["SETUP"].map(stab.groupby("NEWSETUP")["t"].mean()))
    minrun = d["setupgrp"].set_index("SETUP")["MINRUN"].astype(float)

    s = st[st["SETUP"].notna()].copy()
    s["visits"] = s["lots_per_week"] * s["sampling"]
    avail = (g["qty"] * WEEK_MIN * (1 - g["down_frac"]))

    fam = s.groupby("STNFAM").agg(
        n_setup_types=("SETUP", "nunique"), visits_per_week=("visits", "sum"), setup_min=("su_time", "mean"))
    fam["minrun"] = s.groupby("STNFAM")["SETUP"].first().map(minrun).fillna(1)
    fam["qty"] = g["qty"]
    fam["util_no_setup"] = g["util_full"]
    fam["util_campaign"] = fam["util_no_setup"] + fam["visits_per_week"] / fam["minrun"] * fam["setup_min"] / avail
    fam["util_worst"] = fam["util_no_setup"] + fam["visits_per_week"] * fam["setup_min"] / avail

    # 셋업 종류별로 장비를 전용할 때 필요한 대수 (셋업 시간 0 가정)
    per_tool_avail = WEEK_MIN * (1 - g["down_frac"])
    need = (s.groupby(["STNFAM", "SETUP"])["load_full"].sum()
            / s.groupby(["STNFAM", "SETUP"])["STNFAM"].first().map(per_tool_avail))
    ded = need.groupby(level=0).agg(
        dedicated_tools_needed=lambda x: int(sum(math.ceil(v) for v in x)), fractional_need=("sum"))
    fam = fam.join(ded)
    fam["dedication_feasible"] = fam["dedicated_tools_needed"] <= fam["qty"]

    run = json.loads((RUNS / f"greedy_seed0_180days_{model}_fifo.json").read_text())
    mach = pd.DataFrame(run["machines"]).T
    fam["fifo_setup_share"] = mach["setup"].reindex(fam.index)
    fam["fifo_util"] = mach["util"].reindex(fam.index)

    cols = ["qty", "n_setup_types", "minrun", "setup_min", "util_no_setup", "util_campaign", "util_worst",
            "fractional_need", "dedicated_tools_needed", "dedication_feasible", "fifo_setup_share", "fifo_util"]
    fam = fam[cols].sort_values("util_worst", ascending=False)
    print(f"== {model}")
    print(fam.round(3).to_string(), "\n")
    fam.to_csv(OUT / f"eda02_setup_load_{model}.csv")
    need.rename("tools_needed").to_csv(OUT / f"eda02_setup_tools_needed_{model}.csv")
