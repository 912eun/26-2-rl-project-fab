"""SMT2020 데이터 로더와 정적 용량 계산.

해석 규칙은 PySCFabSim(simulation/classes.py, file_instance.py, instance.py)을 따른다.
- 시간은 모두 '분'으로 통일한다.
- 장비 점유시간(lot당):
    cascade = PartInterval*25            (PartInterval이 있으면, 낱장 캐스케이딩)
            | BatchInterval              (BatchInterval이 있으면)
            | 처리시간 평균               (per_piece면 x25)
    점유 = cascade + (load + unload, 단 장비가 캐스케이딩(STNCAP==2)이면 0)
- 배치 공정(per_batch)은 배치당 점유를 배치 lot 수로 나눈다.
- 샘플링(StepPercent) 비율만큼 부하를 곱한다. 리워크(0.5~2%)는 무시한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PIECES = 25
WEEK_MIN = 7 * 24 * 60
UNIT_MIN = {"sec": 1 / 60, "s": 1 / 60, "min": 1.0, "hr": 60.0, "day": 1440.0}


def to_min(value, unit) -> float:
    if pd.isna(value):
        return np.nan
    return float(value) * UNIT_MIN[str(unit)]


def read(model: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / model / name, sep="\t", dtype=str, keep_default_na=False)
    df = df.loc[:, [c for c in df.columns if c.upper() != "IGNORE"]]
    return df.replace("", np.nan)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_model(model: str) -> dict[str, pd.DataFrame]:
    d = {}
    d["tool"] = read(model, "tool.txt.1l").dropna(subset=["STNFAM"])
    d["part"] = read(model, "part.txt")
    d["order"] = read(model, "order.txt")
    d["wip"] = read(model, "WIP.txt")
    routes = []
    for f in sorted((RAW / model).glob("route_*.txt")):
        r = read(model, f.name).dropna(subset=["STEP"])
        r["ROUTEFILE"] = f.name
        routes.append(r)
    d["route"] = pd.concat(routes, ignore_index=True)
    for k in ["downcal", "pmcal", "attach", "setup", "setupgrp", "fromto"]:
        d[k] = read(model, f"{k}.txt")
    return d


def release_rates(d) -> pd.DataFrame:
    """제품(route)별 주당 투입 lot 수. 일반 lot과 hot lot을 함께 센다."""
    o = d["order"].copy()
    interval = [to_min(v, u) for v, u in zip(num(o["REPEAT"]), o["RUNITS"])]
    o["lots_per_week"] = num(o["LOTSPERRPT"]).fillna(1) * WEEK_MIN / np.array(interval)
    o = o.merge(d["part"][["PART", "ROUTEFILE"]], on="PART")
    o["is_hot"] = o["HOTLOT"].str.lower().eq("yes") | o["LOT"].str.startswith("Hot")
    return o[["LOT", "PART", "ROUTEFILE", "PRIOR", "is_hot", "lots_per_week", "START", "DUE"]]


def step_occupancy(d) -> pd.DataFrame:
    """route 단계별 장비 점유시간(분/lot). 배치 공정은 최대·최소 배치 두 경우를 계산."""
    r = d["route"].copy()
    tool = d["tool"].set_index("STNFAM")
    lt = pd.Series([to_min(v, u) for v, u in zip(num(tool["LTIME"]), tool["LTUNITS"])], index=tool.index).fillna(0)
    ult = pd.Series([to_min(v, u) for v, u in zip(num(tool["ULTIME"]), tool["ULTUNITS"])], index=tool.index).fillna(0)
    casc_tool = num(tool["STNCAP"]).eq(2)

    out = []
    for _, s in r.iterrows():
        f = s["STNFAM"]
        per = s["PTPER"]
        ptime = to_min(num(pd.Series([s["PTIME"]]))[0], s["PTUNITS"])
        part_int = num(pd.Series([s.get("PartInterval")]))[0]
        batch_int = num(pd.Series([s.get("BatchInterval")]))[0]
        if not np.isnan(part_int):
            proc = ptime + to_min(part_int, s["PartIntUnits"]) * (PIECES - 1)
            cascade = to_min(part_int, s["PartIntUnits"]) * PIECES
        else:
            proc = ptime * (PIECES if per == "per_piece" else 1)
            cascade = to_min(batch_int, s["BatchIntUnits"]) if not np.isnan(batch_int) else proc
        occ = cascade + (0 if casc_tool.get(f, False) else lt.get(f, 0) + ult.get(f, 0))
        is_batch = per == "per_batch"
        bmax = int(float(s["BATCHMX"]) // PIECES) if is_batch and pd.notna(s["BATCHMX"]) else 1
        bmin = int(float(s["BATCHMN"]) // PIECES) if is_batch and pd.notna(s["BATCHMN"]) else 1
        samp = num(pd.Series([s.get("StepPercent")]))[0]
        samp = 1.0 if np.isnan(samp) else samp / 100
        out.append(dict(
            ROUTEFILE=s["ROUTEFILE"], STEP=int(float(s["STEP"])), DESC=s["DESC"], STNFAM=f,
            area=s.get("DESC", "").split("_", 1)[-1] if isinstance(s.get("DESC"), str) else "",
            per=per, proc_min=proc, occ_per_run_min=occ, is_batch=is_batch,
            batch_min_lots=max(bmin, 1), batch_max_lots=max(bmax, 1), sampling=samp,
            setup=s.get("SETUP"), rework=num(pd.Series([s.get("REWORK")]))[0],
        ))
    o = pd.DataFrame(out)
    o["occ_lot_fullbatch"] = o["occ_per_run_min"] / o["batch_max_lots"] * o["sampling"]
    o["occ_lot_minbatch"] = o["occ_per_run_min"] / o["batch_min_lots"] * o["sampling"]
    return o


def downtime_fraction(d, wafers_per_tool_min: pd.Series) -> pd.DataFrame:
    """장비군별 비가용 시간 비율(고장 + 달력 PM + 매수 기반 PM)의 근사값."""
    tool = d["tool"][["STNFAM", "STNGRP"]]
    dc = d["downcal"].set_index("DOWNCALNAME")
    pc = d["pmcal"].set_index("PMCALNAME")
    rows = []
    for _, a in d["attach"].dropna(subset=["CALNAME"]).iterrows():
        fams = tool.loc[tool["STNGRP"].eq(a["RESNAME"]) if a["RESTYPE"] == "stngrp" else tool["STNFAM"].eq(a["RESNAME"]), "STNFAM"]
        for f in fams:
            if a["CALTYPE"] == "down":
                c = dc.loc[a["CALNAME"]]
                mttf, mttr = to_min(num(pd.Series([c["MTTF"]]))[0], c["MTTFUNITS"]), to_min(num(pd.Series([c["MTTR"]]))[0], c["MTTRUNITS"])
                rows.append((f, "breakdown", mttr / (mttf + mttr)))
            else:
                c = pc.loc[a["CALNAME"]]
                mttr = to_min(num(pd.Series([c["MTTR"]]))[0], c["MTTRUNITS"])
                mtbpm = num(pd.Series([c["MTBPM"]]))[0]
                if c["MTBPMUNITS"] == "pieces":
                    rate = wafers_per_tool_min.get(f, 0) / mtbpm  # PM 횟수/분
                    rows.append((f, "pm_pieces", rate * mttr))
                else:
                    period = to_min(mtbpm, c["MTBPMUNITS"])
                    rows.append((f, "pm_calendar", mttr / (period + mttr)))
    df = pd.DataFrame(rows, columns=["STNFAM", "kind", "frac"])
    return df.pivot_table(index="STNFAM", columns="kind", values="frac", aggfunc="sum").fillna(0)


def static_capacity(model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """장비군별 정적 부하율. 반환: (장비군 표, 단계 표, 투입 표)."""
    d = load_model(model)
    rel = release_rates(d)
    rate = rel.groupby("ROUTEFILE")["lots_per_week"].sum()
    st = step_occupancy(d)
    st["lots_per_week"] = st["ROUTEFILE"].map(rate)
    st["load_full"] = st["lots_per_week"] * st["occ_lot_fullbatch"]
    st["load_min"] = st["lots_per_week"] * st["occ_lot_minbatch"]
    st["wafers_per_week"] = st["lots_per_week"] * PIECES * st["sampling"]

    tool = d["tool"].set_index("STNFAM")
    qty = num(tool["STNQTY"])
    g = st.groupby("STNFAM").agg(
        load_full=("load_full", "sum"), load_min=("load_min", "sum"),
        wafers_per_week=("wafers_per_week", "sum"), n_steps=("STEP", "size"),
        is_batch=("is_batch", "any"), has_setup=("setup", lambda x: x.notna().any()),
    )
    g["STNGRP"] = tool["STNGRP"]
    g["qty"] = qty
    wafer_rate = g["wafers_per_week"] / g["qty"] / WEEK_MIN
    down = downtime_fraction(d, wafer_rate).reindex(g.index).fillna(0)
    g = g.join(down)
    g["down_frac"] = down.sum(axis=1).clip(upper=0.95)
    avail_min = g["qty"] * WEEK_MIN * (1 - g["down_frac"])
    g["util_full"] = g["load_full"] / avail_min   # 가용시간 대비 부하율 (배치 꽉 채움)
    g["util_min"] = g["load_min"] / avail_min     # 최소 배치로 돌릴 때
    return g.sort_values("util_full", ascending=False), st, rel
