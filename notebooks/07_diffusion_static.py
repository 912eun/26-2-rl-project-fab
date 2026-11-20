"""확산로(Diffusion) 정적 데이터 정리: 툴셋, 패밀리(공정+제품), 배치 크기, 처리 시간, 도착률, 부하, CQT.
읽기 전용. 결과는 표준 출력으로만 낸다. (2026-11-13, 확산로 설계 아티팩트용)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from src.smt2020 import load_model, release_rates, num, to_min

for model in ["SMT2020_HVLM", "SMT2020_LVHM"]:
    d = load_model(model)
    rel = release_rates(d)
    perweek = rel.groupby("ROUTEFILE")["lots_per_week"].sum()
    r = d["route"].copy()
    tool = d["tool"].set_index("STNFAM")
    r["STEPn"] = num(r["STEP"]).astype(int)
    diff = r[r["STNFAM"].str.startswith("Diffusion", na=False)].copy()
    print(f"\n===== {model}: diffusion steps {len(diff)}, per {diff.PTPER.value_counts().to_dict()}, SETUP nonnull {diff.SETUP.notna().sum()}")
    diff["bmin"] = (num(diff.BATCHMN) // 25).astype(int)
    diff["bmax"] = (num(diff.BATCHMX) // 25).astype(int)
    diff["pt_h"] = [to_min(v, u) / 60 for v, u in zip(num(diff.PTIME), diff.PTUNITS)]
    diff["pt2_h"] = [to_min(v, u) / 60 for v, u in zip(num(diff.PTIME2), diff.PTUNITS)]
    diff["lots_wk"] = diff.ROUTEFILE.map(perweek)
    diff["samp"] = num(diff.StepPercent).fillna(100) / 100
    # CQT: 앞 공정 행에 STEP_CQT(목표 확산 step), CQT
    cq = r[r.STEP_CQT.notna()][["ROUTEFILE", "STEPn", "DESC", "STNFAM", "STEP_CQT", "CQT", "CQTUNITS"]].copy()
    cq["target"] = num(cq.STEP_CQT).astype(int)
    cq = cq.merge(r[["ROUTEFILE", "STEPn", "STNFAM", "DESC"]].rename(columns={"STEPn": "target", "STNFAM": "tfam", "DESC": "tdesc"}), on=["ROUTEFILE", "target"])
    cqd = cq[cq.tfam.str.startswith("Diffusion")]
    print("CQT rows into diffusion:\n", cqd[["ROUTEFILE", "STEPn", "DESC", "STNFAM", "target", "tdesc", "CQT", "CQTUNITS"]].to_string(index=False))
    rows = []
    for fam, g in diff.groupby("STNFAM"):
        n = int(float(tool.loc[fam, "STNQTY"]))
        lt = to_min(num(pd.Series([tool.loc[fam, "LTIME"]]))[0], tool.loc[fam, "LTUNITS"]) if pd.notna(tool.loc[fam, "LTIME"]) else 0
        ult = to_min(num(pd.Series([tool.loc[fam, "ULTIME"]]))[0], tool.loc[fam, "ULTUNITS"]) if pd.notna(tool.loc[fam, "ULTIME"]) else 0
        load_full = load_m1 = load_min = 0
        fams = []
        for desc, gg in g.groupby("DESC"):
            for rf, x in gg.groupby("ROUTEFILE"):
                visits = len(x)
                lam = x.lots_wk.iloc[0] * x.samp.sum() / 168  # lot/h (fab)
                s = x.iloc[0]
                occ = s.pt_h + (lt + ult) / 60
                load_full += lam * occ / s.bmax
                load_m1 += lam * occ / max(s.bmax - 1, s.bmin)
                load_min += lam * occ / s.bmin
                fams.append(f"{desc.split('_')[0]}/{rf.replace('route_','r').replace('.txt','')} v{visits} ({s.bmin}~{s.bmax}, {s.pt_h:.2f}h±{s.pt2_h:.2f}, fab {lam:.3f} lot/h, per-tool {lam/n:.3f})")
        rows.append((fam, n, load_full / n, load_m1 / n, load_min / n, lt, ult, fams))
    for fam, n, a, b, c, lt, ult, fams in sorted(rows, key=lambda x: -x[2]):
        print(f"{fam:18s} n={n:2d} load full {a:.2f} / max-1 {b:.2f} / min {c:.2f}  LT {lt} ULT {ult}")
        for f in fams: print("    ", f)
