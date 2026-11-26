"""EDA 06 확인 실험: 미래를 모르는 rollout이 cov 규칙을 이기는가.

앞선 실행에서 시나리오 수를 3에서 5로 바꾸며 값을 봤으므로, 그 결과로 판정하지 않고 새로 돌린다.
조건은 실행 전에 고정: 시나리오 10개, 8시간, 고장 시드 1~3(앞서 쓰지 않은 값), 시험 기록 시드 2, 3, 장비군 3개.
되먹임 판(IMPLANT_FEEDBACK=1)에서 돌리며 기준 규칙의 파라미터는 그 판의 tune.json을 쓴다.
판정: 사전 기준대로 짝지은 주간 차이의 평균이 −3% 이하이고 95% 신뢰구간이 0을 포함하지 않으면 온라인 몫이 있다고 본다.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import run_implant_replay as R  # noqa: E402

BASE = {f: tuple(v["best"]["cov"]) for f, v in json.loads((R.OUT / "tune.json").read_text())["1.0_True"].items()}


def job(j):
    f, s, ss = j
    r = R.rollout_job((f, s, BASE[f], "online", 8 * 3600, 1.0, ss, 10))
    b = R.one((f, s, BASE[f], 1.0, ss, True))
    return f, s, ss, r[1], r[2], b[1], b[2], r[3]


if __name__ == "__main__":
    jobs = [(f, s, ss) for f in BASE for s in (2, 3) for ss in (1, 2, 3)]
    out, pooled = [], {}
    with Pool(6) as p:
        for f, s, ss, m, w, bm, bw, sec in p.map(job, jobs):
            print(f, s, ss, round(m["delay_h"], 3), round(bm["delay_h"], 3), f"{sec/60:.1f}분", flush=True)
            out.append(dict(family=f, seed=s, sim=ss, rollout=m, base=bm))
            pooled.setdefault(f, []).extend((w[k] - bw[k]) / bw[k] * 100 for k in w if k in bw)
    summary = {}
    for f, d in pooled.items():
        rows = [x for x in out if x["family"] == f]
        ratio = sum(x["rollout"]["delay_h"] for x in rows) / sum(x["base"]["delay_h"] for x in rows) - 1
        summary[f] = dict(weekly=R.ci(d), ratio_pct=ratio * 100, n_weeks=len(d),
                          setup_rollout=sum(x["rollout"]["setup_ratio"] for x in rows) / len(rows),
                          setup_base=sum(x["base"]["setup_ratio"] for x in rows) / len(rows))
        print(f, summary[f], flush=True)
    (R.OUT / "confirm.json").write_text(json.dumps(dict(summary=summary, runs=out), indent=1))
