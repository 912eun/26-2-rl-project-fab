"""공장 최종 검증(E6) 결과 표. docs/EXPERIMENT_PLAN.md 6-4절.

공장 시드 4·5·6, 2년(통계는 1년 뒤 리셋 이후 = 2년째). 정책 실행 폴더 이름의 규칙 부분(mode_name)을 받는다.
주 지표: implant 장비군별 대기(lot당, 장비 앞 대기). 보조: implant X-factor, 공장 일반 lot 평균 CT, 납기 준수, 2년째 처리량.
비교: cov(covfb)와 같은 공장 시드끼리 짝지은 차이. 시드 3개라 방향만 본다.

예: python3 experiments/fab_report.py --runs lssu covfb rl_dqn_noup_v4_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAS = ["Implant_128", "Implant_132", "Implant_91"]
PROC_H = {"Implant_128": 1.14 * 25 / 60, "Implant_132": 1.05 * 25 / 60, "Implant_91": 1.242 * 25 / 60}  # 낱장 25장 연속 처리 시간(route PTIME)


def one(name, seed):
    d = ROOT / f"results/implant_setup/SMT2020_HVLM_fifo_{name}_Demand_730d_s{seed}"
    files = list((d / "greedy").glob("*.json"))
    if not (d / "run_info.txt").exists() or not files:
        return None
    res = json.loads(files[0].read_text())
    lots = pd.DataFrame(res["lots"]).T
    n1 = lots["throughput_one_year"].astype(float)
    reg = ~lots.index.str.contains("Hot")
    w = n1[reg]
    r = dict(run=name, seed=seed, throughput_y2=float(n1.sum()),
             ct_regular_d=float((lots.loc[reg, "ACT"].astype(float) * w).sum() / w.sum()),
             ontime_regular=float(lots.loc[reg, "on_time"].astype(float).sum() / w.sum() * 100))
    info = json.loads((d / "implant_info.json").read_text())
    for g in GAS:
        n, s = info["lot_waiting_at_machine"][g]
        wh = s / max(n, 1) / 3600
        r[f"wait_{g}"] = wh
        r[f"xf_{g}"] = (wh + PROC_H[g]) / PROC_H[g]
    tot_n = sum(info["lot_waiting_at_machine"][g][0] for g in GAS)
    r["wait_all"] = sum(info["lot_waiting_at_machine"][g][1] for g in GAS) / tot_n / 3600
    if "hold" in info and "rl_calls" in info["hold"]:
        h = info["hold"]
        r["rl_switch_share"] = h["rl_switch"] / max(h["rl_calls"], 1)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["lssu", "covfb"])
    p.add_argument("--base", default="covfb")
    p.add_argument("--seeds", nargs="+", type=int, default=[4, 5, 6])
    a = p.parse_args()
    rows = [r for n in a.runs for s in a.seeds if (r := one(n, s))]
    df = pd.DataFrame(rows)
    cols = ["wait_all"] + [f"wait_{g}" for g in GAS] + [f"xf_{g}" for g in GAS] + ["ct_regular_d", "ontime_regular", "throughput_y2"]
    print("공장 2년째 통계, 공장 시드", a.seeds)
    print(df.groupby("run", sort=False)[cols].mean().T.round(3).to_string())
    base = df[df.run == a.base].set_index("seed")
    for n in a.runs:
        if n == a.base:
            continue
        cur = df[df.run == n].set_index("seed")
        com = cur.index.intersection(base.index)
        print(f"\n{n} − {a.base} (같은 공장 시드끼리, 시드 {list(com)})")
        for c in cols:
            diff = cur.loc[com, c] - base.loc[com, c]
            rel = (cur.loc[com, c] / base.loc[com, c] - 1) * 100
            unit = "%p" if c == "ontime_regular" else ""
            extra = "" if c.startswith(("ontime", "throughput")) else f"  ({', '.join(f'{x:+.1f}%' for x in rel)})"
            print(f"  {c:18s} 평균 차이 {diff.mean():+.3f}{unit}  시드별 {', '.join(f'{x:+.3f}' for x in diff)}{extra}")


if __name__ == "__main__":
    main()
